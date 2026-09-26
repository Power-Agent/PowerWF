from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from b2g_agent.collaboration.models import (
    EngineerRole,
    MediatedReply,
    MediatorPlan,
    ResearchScenario,
    ScenarioParameters,
    SimulationRun,
    SimulationScope,
)
from b2g_agent.collaboration.scenario import (
    allowed_parameter_descriptions,
    compact_case_context,
    role_label,
    scenario_title,
)


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class B2GMediator:
    """LLM mediator with an optional deterministic parser for library tests.

    The API key is read server-side only. The browser never receives it, and no
    key or authorization header is written to session artifacts. The web app
    enables required-LLM mode, so provider failures are surfaced instead of
    silently switching to the deterministic parser.
    """

    def __init__(self, *, client: Any | None = None, model: str | None = None) -> None:
        _load_local_env()
        self.model = model or os.getenv("B2G_MODEL") or os.getenv("B2G_CHAT_MODEL") or "gpt-4.1-mini"
        self.enabled = os.getenv("B2G_LLM_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
        self.require_llm = os.getenv("B2G_REQUIRE_LLM", "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        self._client = client if client is not None else self._build_client()
        self.last_backend = f"openai:{self.model}" if self._client is not None else "not-configured"
        self.last_error: str | None = None

    @property
    def configured(self) -> bool:
        return self._client is not None and self.enabled

    def plan_turn(
        self,
        *,
        text: str,
        scenario_id: ResearchScenario,
        user_role: EngineerRole,
        counterpart_role: EngineerRole,
        parameters: ScenarioParameters,
        recent_messages: list[dict[str, Any]],
    ) -> MediatorPlan:
        if self.configured:
            try:
                plan = self._call_json(
                    MediatorPlan,
                    system=_planning_prompt(user_role, counterpart_role, scenario_id),
                    user=json.dumps(
                        {
                            "message": text,
                            "current_case": parameters.model_dump(mode="json"),
                            "recent_conversation": recent_messages[-8:],
                        },
                        ensure_ascii=False,
                    ),
                )
                self.last_backend = f"openai:{self.model}"
                self.last_error = None
                return plan
            except Exception as exc:  # pragma: no cover - network/provider dependent
                self.last_error = exc.__class__.__name__
                if self.require_llm:
                    raise RuntimeError(
                        "The configured LLM API request failed. Check your API key, model access, "
                        "network connection, and provider quota; B2G-Agent did not use an offline fallback."
                    ) from exc
        elif self.require_llm:
            raise RuntimeError(
                "B2G-Agent requires your LLM API. Add a valid OPENAI_API_KEY and keep "
                "B2G_LLM_ENABLED=true."
            )
        self.last_backend = "deterministic-parser"
        return _local_plan(text, user_role, counterpart_role, scenario_id)

    def compose_reply(
        self,
        *,
        original_text: str,
        scenario_id: ResearchScenario,
        user_role: EngineerRole,
        counterpart_role: EngineerRole,
        parameters: ScenarioParameters,
        plan: MediatorPlan,
        run: SimulationRun | None,
    ) -> MediatedReply:
        if self.configured and self.last_backend.startswith("openai:"):
            try:
                reply = self._call_json(
                    MediatedReply,
                    system=_reply_prompt(user_role, counterpart_role, scenario_id),
                    user=json.dumps(
                        {
                            "original_message": original_text,
                            "mediator_plan": plan.model_dump(mode="json"),
                            "current_case": parameters.model_dump(mode="json"),
                            "simulation": run.model_dump(mode="json") if run else None,
                        },
                        ensure_ascii=False,
                    ),
                )
                self.last_error = None
                return reply
            except Exception as exc:  # pragma: no cover - network/provider dependent
                self.last_error = exc.__class__.__name__
                if self.require_llm:
                    raise RuntimeError(
                        "The configured LLM API request failed while composing the mediated reply; "
                        "B2G-Agent did not use an offline fallback."
                    ) from exc
                self.last_backend = "deterministic-parser"
        elif self.require_llm:
            raise RuntimeError(
                "B2G-Agent requires your LLM API to compose the mediated reply."
            )
        return _local_reply(user_role, counterpart_role, parameters, plan, run, scenario_id)

    def _call_json(
        self,
        response_model: type[ResponseModel],
        *,
        system: str,
        user: str,
    ) -> ResponseModel:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("The LLM returned an empty response.")
        return response_model.model_validate_json(content)

    def _build_client(self) -> Any | None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        placeholder_prefixes = ("your_", "replace_", "example_", "<")
        if not self.enabled or not api_key or api_key.lower().startswith(placeholder_prefixes):
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None
        return OpenAI(api_key=api_key)


def _planning_prompt(
    user_role: EngineerRole,
    counterpart_role: EngineerRole,
    scenario_id: ResearchScenario,
) -> str:
    schema = json.dumps(MediatorPlan.model_json_schema(), ensure_ascii=False)
    allowed = json.dumps(allowed_parameter_descriptions(scenario_id), ensure_ascii=False)
    return f"""
You are the planning layer of B2G-Agent, a simulation-grounded mediator between a
{role_label(user_role)} and a {role_label(counterpart_role)}.
The selected research scenario is {scenario_title(scenario_id)}.

Your responsibilities are to understand the user's free-form engineering message,
translate it for the counterpart, and decide whether a validated simulation is needed.
Do not invent simulation results. Do not claim a model was modified or run.

Return JSON only and exactly match this JSON Schema:
{schema}

Rules:
- requested_changes may contain only these keys and value domains: {allowed}
- Use simulation_scope="coupled" for changes that can affect both buildings and the grid.
- Use simulation_scope="grid" only for a purely grid-capacity or connection change.
- Use simulation_scope="none" for discussion, explanations, or clarification.
- Ask one focused clarification question when a material parameter is ambiguous.
- needs_confirmation=true for topology changes, very large capacity changes, or low-confidence interpretations.
- Preserve the user's professional intent; do not optimize away their stated priorities.
- confidence must reflect interpretation confidence, not confidence in unrun physics.
""".strip()


def _reply_prompt(
    user_role: EngineerRole,
    counterpart_role: EngineerRole,
    scenario_id: ResearchScenario,
) -> str:
    schema = json.dumps(MediatedReply.model_json_schema(), ensure_ascii=False)
    return f"""
You are B2G-Agent after the planning and optional simulation stages.
The human is the {role_label(user_role)}. The simulated counterpart is the
{role_label(counterpart_role)}.
The selected research scenario is {scenario_title(scenario_id)}.

Return JSON only and exactly match this JSON Schema:
{schema}

Requirements:
- mediator_message explains the interpretation and evidence in plain language.
- counterpart_message speaks in first person as the counterpart engineer and responds
  to the user's proposal using only supplied case state and simulation evidence.
- next_question advances the joint design toward a feasible, mutually acceptable plan.
- decision_note records a concrete proposal or result; otherwise use null.
- Clearly distinguish user input, assumptions, and computed results.
- Never fabricate a metric. If simulation is null, say that no new simulation was run.
- Avoid pretending the deterministic vertical backend is a calibrated EnergyPlus or OpenDSS result.
""".strip()


def _local_plan(
    text: str,
    user_role: EngineerRole,
    counterpart_role: EngineerRole,
    scenario_id: ResearchScenario,
) -> MediatorPlan:
    normalized = " ".join(text.lower().split())
    changes: dict[str, Any] = {}

    temperature = re.search(r"(\d+(?:\.\d+)?)\s*(?:°?\s*c|摄氏度|度)", normalized)
    if temperature:
        changes["cooling_setpoint_c"] = float(temperature.group(1))
    building_count = re.search(r"(\d+)\s*(?:栋|buildings?|homes?)", normalized)
    if building_count:
        changes["building_count"] = int(building_count.group(1))
    bus = re.search(r"bus[_\s-]?([3-8])|(?:节点|母线)\s*([3-8])", normalized)
    if bus:
        changes["target_bus"] = f"bus_{bus.group(1) or bus.group(2)}"
    transformer = re.search(r"(\d+(?:\.\d+)?)\s*(?:kva|千伏安)", normalized)
    if transformer:
        changes["transformer_capacity_kva"] = float(transformer.group(1))
    line_capacity = re.search(r"(?:line|线路).*?(\d+(?:\.\d+)?)\s*kw", normalized)
    if line_capacity:
        changes["line_capacity_kw"] = float(line_capacity.group(1))
    demand_response = re.search(r"(?:demand response|需求响应|削峰).*?(\d+(?:\.\d+)?)\s*%", normalized)
    if demand_response:
        changes["demand_response_pct"] = float(demand_response.group(1))
    pv = re.search(r"(?:pv|光伏).*?(\d+(?:\.\d+)?)\s*kw", normalized)
    if pv:
        changes["pv_kw_per_building"] = float(pv.group(1))
    dr_target = re.search(
        r"(?:target|commitment|reduce|reduction|响应负荷|削减|承诺).*?(\d+(?:\.\d+)?)\s*kw",
        normalized,
    )
    if dr_target:
        changes["dr_target_kw_per_building"] = float(dr_target.group(1))
    event_window = re.search(
        r"(?:event|事件).*?(\d{1,2})(?::00)?\s*(?:-|to|至)\s*(\d{1,2})(?::00)?",
        normalized,
    )
    if event_window:
        start = int(event_window.group(1))
        end = int(event_window.group(2))
        changes["dr_event_start_hour"] = start
        changes["dr_event_duration_hours"] = (end - start) % 24 or 1
    baseline_adjustment = re.search(
        r"(?:baseline|基线).*?(?:adjust|调整|上调|下调).*?(-?\d+(?:\.\d+)?)\s*%",
        normalized,
    )
    if baseline_adjustment:
        changes["baseline_adjustment_pct"] = float(baseline_adjustment.group(1))
    if "weather-adjusted" in normalized or "weather adjusted" in normalized or "天气修正" in normalized:
        changes["baseline_method"] = "weather_adjusted"
    elif "matched day" in normalized or "匹配日" in normalized:
        changes["baseline_method"] = "matched_day"
    elif "10-day" in normalized or "ten-day" in normalized or "十日平均" in normalized:
        changes["baseline_method"] = "recent_10_day_average"
    if any(word in normalized for word in ["deep retrofit", "深度改造"]):
        changes["retrofit_level"] = "deep"
    elif any(word in normalized for word in ["standard retrofit", "标准改造"]):
        changes["retrofit_level"] = "standard"

    allowed = set(allowed_parameter_descriptions(scenario_id))
    changes = {key: value for key, value in changes.items() if key in allowed}

    finalize = any(word in normalized for word in ["finalize", "final plan", "定稿", "最终方案"])
    should_simulate = bool(changes) or any(
        word in normalized for word in ["simulate", "simulation", "run", "仿真", "模拟", "分析"]
    )
    if finalize:
        intent = "finalize"
        scope = SimulationScope.NONE
    elif should_simulate:
        intent = "simulate"
        grid_only = changes and set(changes) <= {
            "target_bus",
            "transformer_capacity_kva",
            "line_capacity_kw",
        }
        scope = SimulationScope.GRID if grid_only else SimulationScope.COUPLED
    else:
        intent = "discuss"
        scope = SimulationScope.NONE

    return MediatorPlan(
        intent=intent,
        interpreted_intent=(
            f"The {role_label(user_role)} is proposing a scenario change."
            if changes
            else f"The {role_label(user_role)} is sharing a professional view or question."
        ),
        translation_for_counterpart=(
            f"Translate the message into implications relevant to the {role_label(counterpart_role)}."
        ),
        requested_changes=changes,
        simulation_scope=scope,
        needs_confirmation=False,
        assumptions=[],
        confidence=0.55,
    )


def _local_reply(
    user_role: EngineerRole,
    counterpart_role: EngineerRole,
    parameters: ScenarioParameters,
    plan: MediatorPlan,
    run: SimulationRun | None,
    scenario_id: ResearchScenario,
) -> MediatedReply:
    if plan.clarification_question:
        return MediatedReply(
            mediator_message=plan.clarification_question,
            counterpart_message="I will hold my engineering response until that assumption is clear.",
            next_question=plan.clarification_question,
            decision_note=None,
        )

    if run is None:
        return MediatedReply(
            mediator_message=(
                f"I interpreted this as: {plan.interpreted_intent} No new simulation was run. "
                f"Current shared case: {compact_case_context(parameters, scenario_id)}."
            ),
            counterpart_message=(
                f"As the {role_label(counterpart_role)}, I understand the proposal. "
                "Please identify the parameter or acceptance criterion you want us to test next."
            ),
            next_question="Which concrete scenario change should we test against the joint constraints?",
            decision_note=None,
        )

    metrics = run.metrics
    if scenario_id == ResearchScenario.DEMAND_RESPONSE:
        mediator_message = (
            f"I translated the proposal into the shared demand-response case and ran the {run.backend} backend. "
            f"The agreed baseline peaks at {metrics.baseline_peak_kw_per_building:.2f} kW per building, "
            f"the modeled event delivers {metrics.delivered_reduction_kw_per_building:.2f} kW per building "
            f"({metrics.dr_delivery_pct:.1f}% of the commitment), and post-event rebound is "
            f"{metrics.rebound_pct:.1f}% of the target."
        )
        if metrics.feasible:
            counterpart_message = (
                f"As the {role_label(counterpart_role)}, I can accept this baseline and response commitment "
                "as a candidate for enrollment review, subject to measurement and controls validation."
            )
            next_question = "Should we finalize this commitment or test a more conservative reduction target?"
        else:
            counterpart_message = (
                f"As the {role_label(counterpart_role)}, I cannot approve this demand-response commitment yet. "
                "We should revise the baseline, reduction target, event window, or rebound limit."
            )
            next_question = "Which assumption should we revise first: baseline method, target kW, or rebound limit?"
        return MediatedReply(
            mediator_message=mediator_message,
            counterpart_message=counterpart_message,
            next_question=next_question,
            decision_note=(
                f"Run {run.run_id}: {metrics.dr_delivery_pct:.1f}% DR delivery; "
                f"baseline confidence {metrics.baseline_confidence_score:.1f}/100; "
                f"status={'feasible' if metrics.feasible else 'needs revision'}."
            ),
        )

    mediator_message = (
        f"I translated the proposal into validated scenario parameters and ran the {run.backend} backend. "
        f"The feeder peak is {metrics.feeder_peak_kw:.1f} kW, minimum voltage is "
        f"{metrics.minimum_voltage_pu:.3f} p.u., maximum line loading is "
        f"{metrics.maximum_line_loading_pct:.1f}%, and maximum transformer loading is "
        f"{metrics.maximum_transformer_loading_pct:.1f}%."
    )
    if metrics.feasible:
        counterpart_message = (
            f"As the {role_label(counterpart_role)}, I can accept this as a technically feasible candidate. "
            "I would still like to compare its cost and professional trade-offs with the baseline."
        )
        next_question = "Do you want to accept this candidate or compare one lower-cost alternative?"
    else:
        counterpart_message = (
            f"As the {role_label(counterpart_role)}, I cannot accept this candidate yet because one or more "
            "hard grid constraints are violated. We should adjust demand, connection, or capacity."
        )
        next_question = "Which lever should we test next: building flexibility, retrofit, connection bus, or capacity?"
    return MediatedReply(
        mediator_message=mediator_message,
        counterpart_message=counterpart_message,
        next_question=next_question,
        decision_note=(
            f"Run {run.run_id}: {'feasible' if metrics.feasible else 'needs revision'}; "
            f"building score {metrics.building_satisfaction_score:.1f}, grid score {metrics.grid_reliability_score:.1f}."
        ),
    )


def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parents[3] / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        if not env_path.exists():
            return
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    else:
        load_dotenv(env_path)
