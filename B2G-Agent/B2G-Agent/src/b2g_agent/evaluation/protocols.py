"""Read and validate formal protocols without creating runs or API clients."""
import json

from b2g_agent.evaluation.paper_metrics import (
    QUALITY, COST, MECHANISM_ROLES, MECHANISM_VERSIONS,
    AGENT_METRICS_PROTOCOL, agent_metric_keys,
)

CORE_METHODS = ("full_multi_agent", "same_llm_single_agent")
METHODS = (*CORE_METHODS, *MECHANISM_ROLES)


def load_formal_protocol(case, repo_root, method="full_multi_agent"):
    if method not in METHODS:
        raise ValueError(f"Unsupported formal method: {method}")
    folder = case.case_dir / "evaluation"
    name = (f"{method}_protocol.json" if method in MECHANISM_ROLES else
            "baseline_protocol.json" if method == "same_llm_single_agent" else "proposed_protocol.json")
    protocol = json.loads((folder / name).read_text(encoding="utf-8"))
    if protocol["method"] != method:
        raise ValueError("Protocol method mismatch")
    if (protocol["quality_metrics"] != list(QUALITY)
            or protocol["agent_metrics"] != agent_metric_keys(method)
            or protocol["cost_metrics"] != list(COST)):
        raise ValueError("Formal protocol metric selection mismatch")
    if method in MECHANISM_ROLES:
        proposed = json.loads((folder / "proposed_protocol.json").read_text(encoding="utf-8"))
        exclusions = {"method", "protocol_id", "episodes", "episode_selection", "agent_metrics", "agent_metrics_protocol_id", "scoring_protocol_id"}
        shared = lambda data: {k:v for k,v in data.items() if k not in exclusions}
        scoring = json.loads((folder / "scoring.json").read_text(encoding="utf-8"))
        if shared(protocol) != shared(proposed):
            raise ValueError("Mechanism settings must match the proposed method")
        if (protocol["episodes"] != 200 or protocol.get("episode_selection") != "ascending_episode_id_all_200"
                or protocol["protocol_id"] != f"{method}-luna-full200-12-v{MECHANISM_VERSIONS[method]}"
                or protocol.get("agent_metrics_protocol_id") != AGENT_METRICS_PROTOCOL
                or protocol.get("scoring_protocol_id") != scoring["scoring_protocol_id"]):
            raise ValueError("Mechanism protocol or metric definitions mismatch")
    embedding = json.loads((repo_root / "knowledge/retrieval_config.json").read_text(encoding="utf-8"))["embedding"]
    for field, key in (("embedding_model", "model_id"), ("embedding_revision", "revision"), ("embedding_dimension", "dimension")):
        if protocol[field] != embedding[key]:
            raise ValueError("The embedding release differs from the frozen evaluation protocol")
    return protocol
