"""Reviewer v4: full diagnosis and release gates, no corrective dispatch.

The inherited review method, LLM schema, failure memory and shape-repair budget
are identical to Proposed. Only the pipeline's review-to-repair channel is off.
"""
from b2g_agent.agents.reviewer import ReviewerAgent


class NoCorrectiveFeedbackReviewer(ReviewerAgent):
    corrective_feedback_enabled = False
