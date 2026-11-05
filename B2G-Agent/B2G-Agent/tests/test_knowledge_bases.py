from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from b2g_agent.agents.retrieval import DeterministicFusionRetriever


REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_ROOT = REPO_ROOT / "knowledge"
DOMAINS = ("building", "grid", "cross_domain")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(domain: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (KNOWLEDGE_ROOT / domain / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def test_knowledge_manifest_matches_structured_outputs() -> None:
    manifest = json.loads((KNOWLEDGE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["knowledge_release"] == "case-derived-kb-v3"
    assert manifest["raw_asset_count"] == 52
    assert manifest["raw_asset_bytes"] > 1_000_000
    assert manifest["asset_lock_sha256"] == _sha256(
        KNOWLEDGE_ROOT / "sources" / "asset_lock.json"
    )
    for domain in DOMAINS:
        path = KNOWLEDGE_ROOT / domain / "records.jsonl"
        records = _records(domain)
        assert len(records) == manifest["outputs"][domain]["record_count"]
        assert _sha256(path) == manifest["outputs"][domain]["sha256"]


def test_records_are_source_linked_and_exclude_formal_case_answers() -> None:
    registry = json.loads(
        (KNOWLEDGE_ROOT / "sources" / "source_registry.json").read_text(encoding="utf-8")
    )
    lock = json.loads(
        (KNOWLEDGE_ROOT / "sources" / "asset_lock.json").read_text(encoding="utf-8")
    )["assets"]
    source_ids = {source["source_id"] for source in registry["sources"]}
    all_ids: list[str] = []
    source_coverage: Counter[str] = Counter()
    forbidden = re.compile(
        r"\bep-\d{3}\b|\bbem(?:_|[- ]?)\d{2}\b|contract_oracles|numeric_oracles|groundtruth",
        re.IGNORECASE,
    )
    for domain in DOMAINS:
        for item in _records(domain):
            all_ids.append(str(item["record_id"]))
            assert item["domain"] == domain
            assert item["source_refs"]
            assert not forbidden.search(json.dumps(item, ensure_ascii=False))
            for ref in item["source_refs"]:
                assert ref["source_id"] in source_ids
                assert ref["asset_id"] in lock
                assert ref["sha256"] == lock[ref["asset_id"]]["sha256"]
                source_coverage[ref["source_id"]] += 1
    assert len(all_ids) == len(set(all_ids))
    assert source_coverage.keys() >= {
        "tesp",
        "urbanopt",
        "boptest",
        "citylearn",
        "ochre",
        "brick",
        "saref",
        "openleadr",
        "simbench",
    }


def test_retriever_loads_structured_records_with_balanced_hashed_evidence() -> None:
    retriever = DeterministicFusionRetriever(KNOWLEDGE_ROOT)
    counts = retriever.document_count_by_domain()
    assert counts == {"building": 180, "cross_domain": 180, "grid": 180}
    evidence = retriever.retrieve_balanced(
        "Map building net load in kW to a distribution bus in MW, then check voltage and line loading."
    )
    assert len(evidence) == 6
    assert Counter(item.source.split(":")[1] for item in evidence) == {
        "cross_domain": 2,
        "building": 2,
        "grid": 2,
    }
    assert all(item.sha256 and re.fullmatch(r"[0-9a-f]{64}", item.sha256) for item in evidence)
    assert any(item.evidence_id == "mapping-kw-to-mw-boundary" for item in evidence)


def test_balanced_records_include_retrieval_gate_metadata() -> None:
    required = {
        "units", "time_scales", "spatial_levels", "applicable_roles",
        "business_types", "directions", "review_status",
    }
    for domain in DOMAINS:
        records = _records(domain)
        assert len(records) == 180
        assert all(required <= set(item) for item in records)
        assert len({str(item["record_id"]) for item in records}) == 180
        assert all("pending" not in str(item["review_status"]) for item in records)


def test_retrieval_weights_are_frozen_without_evaluation_tuning() -> None:
    config = json.loads(
        (KNOWLEDGE_ROOT / "retrieval_config.json").read_text(encoding="utf-8")
    )
    assert config["calibration_status"] == "author_selected_frozen_without_relevance_judgments"
    assert config["future_tuning_required"] is False
    for direction in config["directions"].values():
        for partition in direction.values():
            assert sum(partition.values()) == 1.0


def test_cross_domain_library_contains_executable_case_derived_mappings() -> None:
    mappings = {
        str(item["record_id"]): item
        for item in _records("cross_domain")
        if item["record_type"] == "executable_mapping_rule"
    }
    assert mappings.keys() >= {
        "mapping-kw-to-mw-boundary",
        "mapping-meter-to-grid-connection",
        "mapping-time-grid-alignment",
        "mapping-helics-publication-subscription",
        "mapping-openadr-active-period",
        "mapping-request-to-validated-delivery",
    }
    assert all(item["rules"] and item["source_refs"] for item in mappings.values())
