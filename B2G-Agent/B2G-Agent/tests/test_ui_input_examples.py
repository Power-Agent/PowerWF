"""UI copy must be understandable by the unchanged deterministic agent path.

These checks deliberately make no Live API or physics-success claim.
"""
import json
from pathlib import Path
import re

import pytest

from b2g_agent.agents import ActorAgent, ReviewerAgent, TranslatorAgent
from b2g_agent.case_io import CaseBundle
from b2g_agent.contracts import AgentRole, InteractionContext, TaskFamily
from b2g_agent.contracts.readiness import execution_readiness


REPO_ROOT = Path(__file__).resolve().parents[1]


def ui_examples():
    source = (REPO_ROOT / "src/b2g_agent/web/static/app.js").read_text(encoding="utf-8")
    block = re.search(r"const requestExamples = \{(.*?)\n\};", source, re.S).group(1)
    cases = []
    expected = {
        "distribution_grid_operator": [
            (TaskFamily.PEAK_SHAVING, "winter_peak_01", "16:00", "18:00", "bem_01", 3, 10),
            (TaskFamily.CONGESTION_MANAGEMENT, "congestion_risk_01", "17:00", "19:00", "bem_04", 11, 8),
            (TaskFamily.RENEWABLE_ABSORPTION, "pv_surplus_02", "12:00", "14:00", "bem_09", 29, 6),
        ],
        "building_energy_manager": [
            (TaskFamily.BASELINE_FLEXIBILITY, "shoulder_02", "09:00", "11:00", "bem_02", 5, 5),
            (TaskFamily.MARKET_TRADING, "summer_peak_02", "15:00", "17:00", "bem_07", 21, 8),
            (TaskFamily.BASELINE_FLEXIBILITY, "winter_peak_02", "10:00", "12:00", "bem_10", 33, 4),
        ],
    }
    for role, meanings in expected.items():
        array = re.search(rf"{role}:\s*\[(.*?)\]", block, re.S).group(1)
        examples = [json.loads(match.group(0)) for match in re.finditer(r'"(?:[^"\\]|\\.)*"', array)]
        assert len(examples) == 3
        for index, (example, meaning) in enumerate(zip(examples, meanings)):
            cases.append((role, index, example, meaning))
    return cases


def wording(text, variant):
    if variant == "plain":
        return text
    if variant == "copied_prefix":
        return "e.g., " + text
    if variant == "polite":
        return "Hi, could you please help with this request? " + text + " Thank you."
    if variant == "tentative":
        return "Would it be possible to check the feasibility of this? " + text
    if variant == "urgent":
        return "Please handle this as soon as possible: " + text
    if variant == "question":
        return "Can you assess the following request: " + text.rstrip(".") + "?"
    if variant == "mixed_language":
        return "\u9ebb\u70e6\u5e2e\u6211\u8bc4\u4f30\u4ee5\u4e0b\u8bf7\u6c42，\u6682\u4e0d\u627f\u8bfa\u7269\u7406\u53ef\u884c\u6027：" + text + " \u8c22\u8c22！"
    if variant == "reordered":
        intent, remainder = text.split(": ", 1)
        return remainder.rstrip(".") + ". The purpose is " + intent.lower() + "."
    if variant in {"conversational", "separate_sentences"}:
        intent = text.split(": ", 1)[0].lower()
        window = re.search(r"\d{2}:\d{2}-\d{2}:\d{2}", text).group()
        day = re.search(r"on the (\w+) operating day", text).group(1)
        bem = re.search(r"bem_\d+", text).group()
        bus = re.search(r"bus (\d+)", text).group(1)
        power = re.search(r"(\d+) kW", text).group(1)
        price = re.search(r"\d+ USD/MWh", text)
        price_text = f" The offer price is {price.group()}." if price else ""
        service = "upward reserve" if "reserve" in intent else "load increase" if "renewable" in intent else "load reduction"
        if variant == "conversational":
            return f"Hi! Could you help me assess {intent}? I'd like {power} kW of {service} from {bem} at bus {bus} during {window} on the {day} operating day. All times are America/New_York.{price_text} Thanks!"
        return f"This is for {intent}. Please assess the {service}. The building is {bem} at bus {bus}. The amount is {power} kW. Use {window} on the {day} operating day, in America/New_York time.{price_text}"
    raise AssertionError(variant)


@pytest.mark.parametrize("role,index,text,meaning", ui_examples(), ids=lambda value: str(value)[:45])
@pytest.mark.parametrize("variant", ["plain", "copied_prefix", "polite", "tentative", "urgent", "question", "mixed_language", "reordered", "conversational", "separate_sentences"])
def test_example_and_tone_changes_preserve_execution_fields(role, index, text, meaning, variant):
    case = CaseBundle.default(REPO_ROOT)
    family, day, start, end, bem, bus, power = meaning
    context = InteractionContext(interaction_id=f"ui-copy-{role}-{index}", case_id=case.case_id, origin="interactive_inferred")
    # Retrieval does not determine these explicit source fields; disabling it
    # keeps this input-contract regression small and independent of local models.
    translator = TranslatorAgent(REPO_ROOT / "knowledge", retrieval_enabled=False)
    spec = translator.translate(wording(text, variant), sender=AgentRole(role), context=context)
    assert spec.task_families == [family]
    assert spec.sender_role.value == role
    assert spec.target_bems == [bem] and spec.target_buses == [bus]
    assert spec.operating_day == day
    assert spec.fields["target_power"].value == power
    assert spec.fields["target_power"].unit == "kW"
    if family == TaskFamily.MARKET_TRADING:
        assert spec.fields["offer_price"].value == 60
        assert spec.fields["offer_price"].unit == "USD/MWh"
    assert [(w.start, w.end, w.timezone) for w in spec.windows] == [(start, end, "America/New_York")]
    assert not spec.clarification_required, spec.clarification_question
    assert not execution_readiness(spec, case=case)
    plan = ActorAgent().plan(spec)
    assert plan is not None and plan.valid, plan.validation_errors if plan else "No plan"
    review = ReviewerAgent(case=case, allow_memory_writes=False).review(spec, plan, require_physics_evidence=True)
    assert not any(issue.responsible_agent in {"translator", "human"} for issue in review.issues)
    assert review.routing_decision == "actor"
