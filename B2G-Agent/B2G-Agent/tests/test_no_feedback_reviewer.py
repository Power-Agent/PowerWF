"""Offline control-flow verification; fixtures are never formal case results."""
from pathlib import Path
from copy import deepcopy

import pytest

from b2g_agent.agents import ReviewerAgent
from b2g_agent.agents.no_feedback_reviewer import NoCorrectiveFeedbackReviewer
from b2g_agent.contracts.semantic_audit import SemanticAuditResult
from b2g_agent.orchestration import B2GPipeline
from reviewer_fixtures import categorical_fixture
from test_correction_workflow import ScriptedLLM, PhysicsDouble, assessment, events

ROOT=Path(__file__).resolve().parents[1]


def setup(tmp_path, hook=None, *, episode="ep-001", method="reviewer_no_feedback", **physics):
    llm=ScriptedLLM(hook)
    pipeline=B2GPipeline.default(repo_root=ROOT,llm_client=llm,formal_evaluation=True,method_id=method)
    pipeline.physics_executor=PhysicsDouble(tmp_path,**physics)
    ep=pipeline.case.get_episode(episode)
    pending=pipeline.run_contract_stage(ep.sender_message,episode=ep,sender=ep.start_role)
    return pipeline,pending,llm


def test_reviewer_retains_full_review_implementation_memory_schema_and_budgets(tmp_path):
    pipeline,_,llm=setup(tmp_path)
    baseline=B2GPipeline.default(repo_root=ROOT,llm_client=llm,formal_evaluation=True)
    assert type(pipeline.reviewer) is NoCorrectiveFeedbackReviewer
    assert NoCorrectiveFeedbackReviewer.review is ReviewerAgent.review
    assert NoCorrectiveFeedbackReviewer._semantic_assessment is ReviewerAgent._semantic_assessment
    assert pipeline.reviewer.memory_path==baseline.reviewer.memory_path
    assert pipeline.reviewer.semantic_enabled is True
    assert not pipeline.reviewer.allow_memory_writes
    assert pipeline.maximum_agent_revisions==baseline.maximum_agent_revisions==4
    assert type(pipeline.translator) is type(baseline.translator)
    assert type(pipeline.actor) is type(baseline.actor)


def test_normal_initial_execution_and_reverse_translation_still_release(tmp_path):
    pipeline,pending,llm=setup(tmp_path)
    assert pending.workflow_status=="awaiting_physics"
    completed=pipeline.execute_physics(pending,run_root=tmp_path/"physics-double")
    assert completed.workflow_status=="released"
    assert pipeline.physics_executor.calls==1
    assert llm.counts=={"translator":1,"actor":1,"reviewer":2,"translator_feedback":1}
    assert not any(e["event"]=="reviewer.correction_withheld" for e in events(completed))


def test_same_review_inputs_produce_identical_llm_requests_and_diagnoses(tmp_path):
    _,pending,_=setup(tmp_path)
    requests=[]
    class Capture:
        available=True
        def call(self,model,**kwargs):
            requests.append((model,deepcopy(kwargs)))
            assert model is SemanticAuditResult
            return model.model_validate(categorical_fixture(assessment(),kwargs["user_payload"]["review_phase"]))
    decisions=[]
    for method in ["full_multi_agent","reviewer_no_feedback"]:
        pipeline=B2GPipeline.default(repo_root=ROOT,llm_client=Capture(),formal_evaluation=True,method_id=method)
        decisions.append(pipeline.reviewer.review(pending.task_specification,pending.execution_plan,
            source_text=pending.task_specification.source_text,draft="Identical reviewed draft.").model_dump(mode="json"))
    assert requests[0]==requests[1] and decisions[0]==decisions[1]


@pytest.mark.parametrize("method,translations,actor_calls",[
    ("reviewer_no_feedback",1,1),("full_multi_agent",2,2),
])
def test_forward_semantic_findings_stop_only_the_ablated_feedback_channel(tmp_path,method,translations,actor_calls):
    def hook(agent,n,user,payload):
        if agent=="reviewer" and n==1:return assessment(True)
    pipeline,pending,llm=setup(tmp_path,hook,method=method)
    assert llm.counts["translator"]==translations and llm.counts["actor"]==actor_calls
    assert pending.workflow_status==("blocked" if method=="reviewer_no_feedback" else "awaiting_physics")
    assert pipeline.physics_executor.calls==0
    if method=="reviewer_no_feedback":
        assert pending.reviewer_decision.issues and pending.reviewer_decision.revision_count==0
        assert any(e["event"]=="reviewer.correction_withheld" for e in events(pending))
        assert not any(e["event"]=="reviewer.return" for e in events(pending))


def test_invalid_initial_plan_is_blocked_without_corrective_actor_calls(tmp_path):
    def hook(agent,n,user,payload):
        if agent=="actor" and n==1:
            return {"actions":payload["actions"][:-1]}
    pipeline,pending,llm=setup(tmp_path,hook)
    assert pending.workflow_status=="blocked" and not pending.execution_plan.valid
    assert llm.counts["actor"]==1 and llm.counts["reviewer"]==1
    assert any(i.code=="A-INVALID-PLAN" for i in pending.reviewer_decision.issues)
    assert pipeline.physics_executor.calls==0


def test_failed_physical_evidence_does_not_trigger_corrective_replanning(tmp_path):
    pipeline,pending,llm=setup(tmp_path,failures=1)
    completed=pipeline.execute_physics(pending,run_root=tmp_path/"physics-double")
    assert completed.workflow_status=="blocked"
    assert pipeline.physics_executor.calls==1 and llm.counts["actor"]==1
    assert not completed.physics_evidence.hard_constraints_passed
    assert any(i.code=="E-HARD-CONSTRAINT-VIOLATION" for i in completed.reviewer_decision.issues)
    assert any(e["event"]=="reviewer.correction_withheld" and e["payload"]["direction"]=="physics" for e in events(completed))


def test_reverse_semantic_findings_withhold_release_without_retranslation(tmp_path):
    def hook(agent,n,user,payload):
        if agent=="reviewer" and user["review_phase"]=="receiver_feedback":return assessment(True)
    pipeline,pending,llm=setup(tmp_path,hook)
    completed=pipeline.execute_physics(pending,run_root=tmp_path/"physics-double")
    assert completed.workflow_status=="blocked"
    assert llm.counts["translator_feedback"]==1 and llm.counts["actor"]==1
    assert completed.reviewer_decision.revision_count==0
    assert any(e["event"]=="reviewer.correction_withheld" and e["payload"]["direction"]=="reverse" for e in events(completed))


def test_necessary_sender_clarification_remains_a_legitimate_outcome(tmp_path):
    def hook(agent,n,user,payload):
        if agent=="reviewer":
            return {**assessment(),"clarification_verdict":"required",
                "clarification_basis":"The supplied request is missing a target quantity."}
    pipeline,pending,llm=setup(tmp_path,hook,episode="ep-002")
    assert pending.workflow_status=="needs_clarification"
    assert pending.information_exchange.decision=="needs_clarification"
    assert llm.counts=={"translator":1,"reviewer":1}
    assert pipeline.physics_executor.calls==0


def test_reviewer_format_failures_still_receive_two_shape_repairs_and_remain_failures(tmp_path):
    def hook(agent,n,user,payload):
        if agent=="reviewer":return {**assessment(),"checks":[]}
    llm=ScriptedLLM(hook)
    pipeline=B2GPipeline.default(repo_root=ROOT,llm_client=llm,formal_evaluation=True,method_id="reviewer_no_feedback")
    ep=pipeline.case.get_episode("ep-001")
    with pytest.raises(RuntimeError,match="format-repair"):
        pipeline.run_contract_stage(ep.sender_message,episode=ep,sender=ep.start_role)
    assert llm.counts=={"translator":1,"actor":1,"reviewer":3}
