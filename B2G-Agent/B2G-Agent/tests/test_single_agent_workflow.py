"""Offline baseline isolation and diagnostic regression tests (no API/simulator)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from b2g_agent.agents import ActorAgent, TranslatorAgent
from b2g_agent.case_io import CaseBundle
from b2g_agent.evaluation import InteractionScorer
from b2g_agent.evaluation.agent_metrics import evaluate_agents
from b2g_agent.orchestration import B2GPipeline
from test_correction_workflow import PhysicsDouble
from test_evaluation_methods import _SingleAgentStub

ROOT = Path(__file__).resolve().parents[1]


class BaselineStub(_SingleAgentStub):
    def __init__(self, spec, plan, *, always_invalid=False, first_invalid=False):
        super().__init__(spec, plan)
        self.calls = []
        self.always_invalid, self.first_invalid = always_invalid, first_invalid

    def call(self, model, *, system, user_payload, agent_name):
        self.calls.append((agent_name, user_payload))
        result = super().call(model, system=system, user_payload=user_payload, agent_name=agent_name)
        if agent_name == 'same_llm_single_agent_contract' and (
                self.always_invalid or (self.first_invalid and len(self.calls) == 1)):
            result.plan.actions[-1].depends_on = []
        return result


def baseline(tmp_path, **kwargs):
    case = CaseBundle.default(ROOT)
    episode = case.get_episode('ep-001')
    spec = TranslatorAgent(ROOT / 'knowledge').translate(
        episode.sender_message, episode=episode, sender=episode.start_role)
    llm = BaselineStub(spec, ActorAgent().plan(spec), **kwargs)
    pipeline = B2GPipeline.default(repo_root=ROOT, method_id='same_llm_single_agent',
                                   llm_client=llm, formal_evaluation=True)
    pipeline.maximum_agent_revisions = 2
    pipeline.physics_executor = PhysicsDouble(tmp_path)
    def forbidden(*args, **kw):
        raise AssertionError('Baseline leaked into a role-specific agent')
    pipeline.translator.translate = forbidden
    pipeline.translator.translate_feedback = forbidden
    pipeline.actor.plan = forbidden
    pending = pipeline.run_contract_stage(episode.sender_message, episode=episode, sender=episode.start_role)
    return pipeline, pending, llm


def test_plan_repair_returns_to_single_agent_and_records_first_candidate(tmp_path):
    pipeline, pending, llm = baseline(tmp_path, first_invalid=True)
    assert pending.execution_plan.valid
    assert pending.workflow_status == 'awaiting_physics'
    assert len(llm.calls) == 2
    assert llm.calls[1][1]['validation_feedback']
    assert 'bem_to_bus' in llm.calls[0][1]['public_case_context']
    assert 'target_power' in llm.calls[0][1]['execution_field_contract']
    assert 'winter_peak_01' in llm.calls[0][1]['available_operating_days']
    assert 'comfort_violation_minutes' in llm.calls[0][1]['supported_requested_outputs']
    assert len(llm.calls[0][1]['retrieved_knowledge']) == 6
    events = [json.loads(s) for s in Path(pending.interaction_journal).read_text(encoding='utf-8').splitlines()]
    candidates = [e['payload'] for e in events if e['event'] == 'single_agent.candidate']
    assert len(candidates) == 2 and not candidates[0]['plan']['valid']
    assert any(e['event'] == 'single_agent.revision' for e in events)


@pytest.mark.parametrize('exceptions,failures', [(1, 0), (0, 1)])
def test_physics_repair_never_calls_translator_actor_or_llm_reviewer(tmp_path, exceptions, failures):
    pipeline, pending, llm = baseline(tmp_path)
    pipeline.physics_executor = PhysicsDouble(tmp_path, exceptions=exceptions, failures=failures)
    completed = pipeline.execute_physics(pending, run_root=tmp_path / 'physics')
    assert pipeline.physics_executor.calls == 2
    assert completed.workflow_status == 'released'
    assert completed.information_exchange.sender_received == 'The requested service was accepted with validated completion.'
    assert pipeline.reviewer.llm is None
    assert len(completed.llm_request_hashes) == len(llm.calls)
    assert set(name for name, _ in llm.calls) <= {'same_llm_single_agent_contract', 'same_llm_single_agent_result'}
    assert any(payload.get('validation_feedback') for _, payload in llm.calls)
    module_spec = importlib.util.spec_from_file_location('baseline_return_test', ROOT/'experiments/report_proposed_suite.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    journal = module.inspect_journal(Path(completed.interaction_journal), tmp_path/'reader/transcript.md', 'ep-001-r01')
    events = [json.loads(s) for s in Path(completed.interaction_journal).read_text(encoding='utf-8').splitlines()]
    assert sum(journal['return_routes'].values()) == sum(e['event']=='single_agent.revision' for e in events)
    assert set(journal['return_routes']) == {'single_agent'}


def test_invalid_plan_exhaustion_is_blocked_without_physics(tmp_path):
    pipeline, pending, llm = baseline(tmp_path, always_invalid=True)
    assert len(llm.calls) == 3 and pending.workflow_status == 'blocked'
    assert pipeline.physics_executor.calls == 0


def test_baseline_diagnostics_capture_functions_but_not_fictitious_reviewer(tmp_path):
    pipeline, pending, _ = baseline(tmp_path)
    module_spec = importlib.util.spec_from_file_location('baseline_report_test', ROOT/'experiments/report_proposed_suite.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    journal = module.inspect_journal(Path(pending.interaction_journal), tmp_path/'reader/transcript.md', 'ep-001-r01')
    assert journal['first_specification'] and journal['first_plan']
    record = dict(row=dict(run_id='ep-001-r01', episode_id='ep-001', interaction_journal=pending.interaction_journal), journal=journal)
    metrics = evaluate_agents([record], pipeline.case, InteractionScorer(pipeline.case.case_dir/'evaluation/scoring.json'), method_id='same_llm_single_agent')
    # A pending plan is not a terminal result and must not receive terminal credit.
    assert metrics['metrics']['actor_plan_legality']['value'] == 0
    assert metrics['metrics']['translator_required_field_f1']['value'] > 0
    for name in ('reviewer_first_review_appropriate_handling', 'reviewer_executable_reverse_review_reach', 'reviewer_executable_task_success'):
        assert metrics['metrics'][name]['value'] is None
        assert metrics['metrics'][name]['status'] == 'not_applicable_no_reviewer'
    assert metrics['review_samples_expected'] == 0


def test_verified_physics_overrides_stale_infeasible_tag_but_not_failed_constraints():
    case = CaseBundle.default(ROOT)
    episode, truth = case.get_episode('ep-080'), case.get_groundtruth('ep-080')
    numeric = json.loads((case.case_dir/'groundtruth/numeric_oracles/ep-080.json').read_text(encoding='utf-8'))
    assert episode.difficulty.feasibility == 'infeasible'
    scorer = InteractionScorer(case.case_dir/'evaluation/scoring.json')
    candidate = SimpleNamespace(information_exchange=SimpleNamespace(decision='accepted'),
        physics_evidence=SimpleNamespace(converged=True, hard_constraints_passed=True))
    assert scorer._physical_feasibility(episode, truth, candidate, numeric) == 1
    candidate.physics_evidence.hard_constraints_passed = False
    assert scorer._physical_feasibility(episode, truth, candidate, numeric) == 0


def test_baseline_protocol_matches_proposed_settings():
    directory = ROOT/'cases/ieee33_10bem_15min/evaluation'
    proposed = json.loads((directory/'proposed_protocol.json').read_text(encoding='utf-8'))
    baseline_protocol = json.loads((directory/'baseline_protocol.json').read_text(encoding='utf-8'))
    assert baseline_protocol['agent_metrics'] == []
    assert len(proposed['agent_metrics']) == 9
    assert {k:v for k,v in proposed.items() if k not in {'protocol_id','method','agent_metrics'}} == {
        k:v for k,v in baseline_protocol.items() if k not in {'protocol_id','method','agent_metrics'}}


def test_suite_physics_paths_are_short_and_owned_by_one_suite(tmp_path):
    spec = importlib.util.spec_from_file_location('baseline_storage_test', ROOT/'experiments/run_proposed_suite.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = tmp_path/'outputs/benchmark/20261031T230000Z-baseline'
    proposed = base.with_name('20261031T230000Z')
    cache = module.physics_cache_root(base)
    assert cache == module.physics_cache_root(base)
    assert cache != module.physics_cache_root(proposed)
    assert cache.parent == tmp_path/'outputs/physics/cache'
    assert len(str(cache)) < 259
    protocol = json.loads((ROOT/'cases/ieee33_10bem_15min/evaluation/baseline_protocol.json').read_text(encoding='utf-8'))
    assert module.protocol_environment(protocol, base, 0)['B2G_SIMULATION_OUTPUT_ROOT'] == str(cache)
