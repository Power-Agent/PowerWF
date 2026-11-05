from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from b2g_agent.agents.actor import ActorAgent, PlanCompiler
from b2g_agent.agents.mechanism_ablations import (
    PlainTranslator, UnstructuredActor, UNSTRUCTURED_VERSION,
    ordinary_tool_catalog,
)
from b2g_agent.agents.translator import TranslatorAgent
from b2g_agent.case_io import CaseBundle
from b2g_agent.contracts import TaskSpecification
from b2g_agent.evaluation.methods import EvaluationMethodRegistry
from b2g_agent.evaluation.paper_metrics import MECHANISM_ROLES, agent_metric_keys, paper_metric_count
from b2g_agent.evaluation.protocols import load_formal_protocol
from b2g_agent.orchestration import B2GPipeline
from free_plan_fixtures import render_plan

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))


@pytest.fixture
def case_spec():
    case = CaseBundle.default(ROOT)
    ep = case.get_episode("ep-001")
    spec = TranslatorAgent(ROOT / "knowledge").translate(ep.sender_message, episode=ep, sender=ep.start_role)
    return case, ep, spec


class CapturingLLM:
    available = True
    request_hashes = []

    def __init__(self, spec=None, text=None):
        self.spec, self.text, self.calls = spec, text, []

    def call(self, model, **kwargs):
        self.calls.append(kwargs)
        assert model is TaskSpecification
        return self.spec.model_copy(deep=True)

    def call_text(self, **kwargs):
        self.calls.append(kwargs)
        return self.text


def test_plain_translation_has_no_semantic_prefill_and_preserves_model_errors(case_spec):
    case, ep, spec = case_spec
    spec.target_buses = [33] * len(spec.target_buses)
    llm = CapturingLLM(spec)
    agent = PlainTranslator(llm=llm, case=case)
    result = agent.translate(ep.sender_message, episode=ep, sender=ep.start_role)
    payload = llm.calls[0]["user_payload"]
    assert payload["source_text"] == ep.sender_message
    assert not {"deterministic_draft", "source_target_pairs", "retrieved_knowledge", "task_families"} & payload.keys()
    assert set(payload["interaction_context"]) == {"interaction_id", "case_id", "origin"}
    assert {"target_power", "time_resolution", "operating_day", "windows", "target_bems", "target_buses"} <= payload["execution_field_contract"].keys()
    assert payload["available_operating_days"] and "winter_peak_01" in payload["available_operating_days"]
    assert not any(isinstance(v, dict) and "value" in v for v in payload["execution_field_contract"].values())
    assert result.target_buses == spec.target_buses
    assert result.retrieval_evidence == []
    assert result.retrieval_trace["elapsed_seconds"] == 0


def test_plain_cannot_silently_use_offline_rule_translation(case_spec):
    case, ep, _ = case_spec
    agent = PlainTranslator(llm=SimpleNamespace(available=False), case=case)
    with pytest.raises(RuntimeError):
        agent.translate(ep.sender_message, episode=ep, sender=ep.start_role)


def test_ordinary_tool_catalog_contains_interfaces_not_task_solutions():
    catalog = ordinary_tool_catalog()
    assert len(catalog) == 14
    assert all(not {"depends_on", "conditional_rule", "input_binding_rule", "exact_action_input_bindings"} & row.keys() for row in catalog)
    assert all("input_fields" in row and "description" in row for row in catalog)


def test_unstructured_plan_retains_declared_cost_and_receives_no_prefilled_values(case_spec):
    _, _, spec = case_spec
    plan = PlanCompiler().compile(spec)
    plan.actions[0].inputs["estimated_cost_units"] = 999
    llm = CapturingLLM(text=render_plan(plan))
    result = UnstructuredActor(llm=llm).plan(spec)
    assert result.valid
    assert result.compiler_version == UNSTRUCTURED_VERSION
    assert result.actions[0].inputs["estimated_cost_units"] == 999
    assert result.normalization_log == []
    payload = llm.calls[0]["user_payload"]
    assert "tools" in payload
    assert not {"tool_capability_graph", "exact_action_input_bindings", "reference_plan"} & payload.keys()


def test_valid_serial_execution_is_not_rejected_for_reference_edge_mismatch(case_spec):
    _, _, spec = case_spec
    plan = PlanCompiler().compile(spec)
    for i, action in enumerate(plan.actions):
        action.depends_on = [plan.actions[i-1].action_id] if i else []
    plan.compiler_version = UNSTRUCTURED_VERSION
    assert PlanCompiler().validate(plan, spec) == []
    # The missing actual data prerequisite must still fail.
    plan.actions[-1].depends_on = []
    assert any(e.startswith("missing_data_dependency:") for e in PlanCompiler().validate(plan, spec))


def test_wrong_unstructured_parameters_are_never_filled_from_spec(case_spec):
    _, _, spec = case_spec
    plan = PlanCompiler().compile(spec)
    next(a for a in plan.actions if a.action_type.value == "map_loads_to_buses").inputs["target_buses"] = [33]
    result = UnstructuredActor(llm=CapturingLLM(text=render_plan(plan))).plan(spec)
    assert not result.valid
    assert "input_binding_mismatch:map_loads_to_buses" in result.validation_errors
    assert next(a for a in result.actions if a.action_type.value == "map_loads_to_buses").inputs["target_buses"] == [33]


def test_unparseable_unstructured_plan_remains_missing(case_spec):
    _, _, spec = case_spec
    assert UnstructuredActor(llm=CapturingLLM(text="Please just perform the analysis.")).plan(spec) is None


@pytest.mark.parametrize("method", MECHANISM_ROLES)
def test_v2_protocols_and_registration_are_isolated(method, case_spec):
    case, _, _ = case_spec
    p = load_formal_protocol(case, ROOT, method)
    assert p["episodes"] == 200 and p["repeats"] == 3
    assert len(agent_metric_keys(method)) == 3 and paper_metric_count(method) == 12
    pipeline = B2GPipeline.default(repo_root=ROOT, method_id=method)
    assert isinstance(pipeline.translator, PlainTranslator) == (method == "plain_translator")
    assert isinstance(pipeline.actor, UnstructuredActor) == (method == "unstructured_actor")
    selected = EvaluationMethodRegistry.load(case.case_dir / "evaluation/methods.json")
    assert set(MECHANISM_ROLES) <= {m.method_id for m in selected.methods}


@pytest.mark.parametrize("method", ["static_reviewer", "generic_reviewer"])
def test_retired_reviewer_versions_are_not_runnable(method, case_spec):
    case, _, _ = case_spec
    with pytest.raises(ValueError, match="Unsupported formal method"):
        load_formal_protocol(case, ROOT, method)
    with pytest.raises(KeyError, match="Unknown evaluation method"):
        B2GPipeline.default(repo_root=ROOT, method_id=method)


def test_full200_runner_selects_all_episodes_once(case_spec):
    import run_mechanism_ablations as runner
    case, _, _ = case_spec
    ids = runner.selected_episodes(case)
    assert ids == [f"ep-{n:03d}" for n in range(1,201)]
    p = load_formal_protocol(case, ROOT, "plain_translator")
    commands = [runner.worker_command("plain_translator", ids[i::4], ROOT / "outputs/controlled-test-not-a-run", i, p) for i in range(4)]
    selected = [c[i+1] for c in commands for i,v in enumerate(c) if v == "--episode-id"]
    assert sorted(selected) == ids
    assert all(c[c.index("--repeats")+1] == "3" for c in commands)


def test_v2_suite_names_keep_all_atomic_record_paths_within_windows_limit():
    import run_mechanism_ablations as runner
    paths = [runner.suite_directory(m, "20261101T000000Z") for m in MECHANISM_ROLES]
    assert len(set(paths)) == len(MECHANISM_ROLES)
    for root in paths:
        longest = root / "lr/0" / ("a"*64) / "i/000001-12345678" / "manifest.json.12345678.tmp"
        assert len(str(longest)) < 259


def test_record_storage_preflight_exercises_atomic_write_without_leaving_fake_calls(tmp_path):
    import run_mechanism_ablations as runner
    runner.check_record_paths(tmp_path, 2)
    assert not list(tmp_path.rglob("*.json"))
    assert not list(tmp_path.rglob("*.tmp"))
