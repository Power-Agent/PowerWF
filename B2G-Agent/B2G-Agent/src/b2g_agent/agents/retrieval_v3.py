from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from b2g_agent.contracts import EvidenceRef
from b2g_agent.observability.journal import traced_stage


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*|[\u4e00-\u9fff]+|\d+(?:\.\d+)?")
EMBEDDING_DIMENSION = 384
SEMANTIC_GROUPS: tuple[tuple[str, ...], ...] = (
    ("curtailment", "curtail", "shed", "reduction", "reduce", "\u524a\u51cf", "\u5f03\u5149"),
    ("precooling", "pre-cooling", "thermal", "inertia", "temperature", "thermostat"),
    ("rebound", "recovery", "post-event"),
    ("comfort", "occupied", "occupancy", "zone"),
    ("voltage", "volt", "vm", "p.u.", "pu", "\u7535\u538b"),
    ("congestion", "overload", "loading", "thermal-limit", "\u963b\u585e"),
    ("reserve", "capacity", "activation", "\u5907\u7528"),
    ("renewable", "solar", "pv", "photovoltaic", "\u5149\u4f0f"),
    ("baseline", "counterfactual", "reference", "\u57fa\u7ebf"),
    ("flexibility", "flexible", "envelope", "dispatch"),
    ("market", "offer", "price", "settlement", "bid"),
    ("building", "bem", "zone", "meter"),
    ("grid", "feeder", "bus", "node", "line", "transformer", "network"),
    ("power", "kw", "mw", "active-power", "demand", "load"),
    ("energy", "kwh", "mwh", "joule"),
    ("reactive", "kvar", "mvar", "power-factor"),
    ("interval", "timeseries", "time-series", "window", "schedule"),
    ("acceptance", "accepted", "commitment", "delivery", "completion"),
)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _semantic_embedding(text: str) -> list[float]:
    """Pinned semantic-feature embedding with no network/model dependency."""

    tokens = _tokens(text)
    token_set = set(tokens)
    vector = [0.0] * EMBEDDING_DIMENSION
    lowered = text.lower()
    for index, aliases in enumerate(SEMANTIC_GROUPS):
        hits = sum(1 for alias in aliases if alias in token_set or alias in lowered)
        if hits:
            vector[index] = 1.0 + math.log1p(hits)
    offset = len(SEMANTIC_GROUPS)
    width = EMBEDDING_DIMENSION - offset
    for token, count in Counter(tokens).items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        slot = offset + int.from_bytes(digest[:4], "big") % width
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[slot] += sign * (1.0 + math.log1p(count))
    return vector


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    domain: str
    text: str
    source: str
    locator: str
    sha256: str
    record_type: str
    units: frozenset[str]
    time_scales: frozenset[str]
    spatial_levels: frozenset[str]
    applicable_roles: frozenset[str]
    business_types: frozenset[str]
    directions: frozenset[str]


class DirectionalFusionRetriever:
    """Direction-aware BM25 + semantic-embedding weighted rank fusion."""

    method_id = "bm25+minilm-dense-embedding+weighted-rrf-v4"

    def __init__(self, knowledge_root: Path) -> None:
        self.knowledge_root = knowledge_root
        self.config = self._load_config()
        self.documents = self._load_documents()
        self._tokenized = {doc.document_id: _tokens(doc.text) for doc in self.documents}
        self._embedding_backend = os.getenv(
            "B2G_EMBEDDING_BACKEND",
            str(self.config["embedding"]["implementation"]),
        ).strip()
        self._embedding_model: Any | None = None
        self._embeddings = self._encode_documents()
        self._documents_by_domain = {
            domain: [doc for doc in self.documents if doc.domain == domain]
            for domain in {doc.domain for doc in self.documents}
        }

    @traced_stage("retrieval.translator")
    def retrieve_for_transfer(
        self,
        query: str,
        *,
        direction: str,
        business_types: Iterable[str] = (),
        time_scales: Iterable[str] = (),
        spatial_levels: Iterable[str] = (),
        roles: Iterable[str] = (),
        per_partition: int = 2,
    ) -> list[EvidenceRef]:
        normalized = _normalize_direction(direction)
        source_domain, target_domain = (
            ("building", "grid") if normalized == "building_to_grid" else ("grid", "building")
        )
        partitions = (("map", "cross_domain"), ("source", source_domain), ("target", target_domain))
        metadata = {
            "business_types": set(business_types),
            "time_scales": set(time_scales),
            "spatial_levels": set(spatial_levels),
            "applicable_roles": set(roles),
        }
        evidence: list[EvidenceRef] = []
        for partition, domain in partitions:
            all_candidates = self._documents_by_domain.get(domain, [])
            candidates = self._gate(all_candidates, metadata)
            weights = self.config["directions"][normalized][partition]
            evidence.extend(
                self._as_evidence(doc)
                for doc in self._rank_documents(
                    query,
                    candidates,
                    sparse_weight=float(weights["bm25"]),
                    dense_weight=float(weights["semantic_embedding"]),
                )[:per_partition]
            )
        return evidence

    def retrieve(
        self,
        query: str,
        *,
        domains: set[str],
        top_k: int = 5,
    ) -> list[EvidenceRef]:
        """Compatibility global retrieval used only by the single-agent baseline."""

        candidates = [doc for doc in self.documents if doc.domain in domains]
        return [
            self._as_evidence(doc)
            for doc in self._rank_documents(
                query, candidates, sparse_weight=0.5, dense_weight=0.5
            )[:top_k]
        ]

    def document_count_by_domain(self) -> dict[str, int]:
        return dict(sorted(Counter(doc.domain for doc in self.documents).items()))

    def retrieve_balanced(
        self,
        query: str,
        *,
        domain_order: tuple[str, ...] = ("cross_domain", "building", "grid"),
        per_domain: int = 2,
    ) -> list[EvidenceRef]:
        """Compatibility entry point for callers without transfer metadata."""

        evidence: list[EvidenceRef] = []
        for domain in domain_order:
            evidence.extend(
                self._as_evidence(doc)
                for doc in self._rank_documents(
                    query,
                    self._documents_by_domain.get(domain, []),
                    sparse_weight=0.5,
                    dense_weight=0.5,
                )[:per_domain]
            )
        return evidence

    def _rank_documents(
        self,
        query: str,
        candidates: list[KnowledgeDocument],
        *,
        sparse_weight: float,
        dense_weight: float,
    ) -> list[KnowledgeDocument]:
        if not candidates:
            return []
        query_tokens = _tokens(query)
        query_vector = self._encode_query(query)
        tokenized = [self._tokenized[doc.document_id] for doc in candidates]
        average_length = sum(map(len, tokenized)) / max(1, len(candidates))
        document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
        sparse: dict[str, float] = {}
        dense: dict[str, float] = {}
        for doc in candidates:
            tokens = self._tokenized[doc.document_id]
            counts = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                df = document_frequency[token]
                inverse_frequency = math.log(1.0 + (len(candidates) - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.5 * (0.25 + 0.75 * len(tokens) / max(1.0, average_length))
                score += inverse_frequency * frequency * 2.5 / denominator
            sparse[doc.document_id] = score
            dense[doc.document_id] = _cosine(query_vector, self._embeddings[doc.document_id])
        sparse_rank = _rank(sparse)
        dense_rank = _rank(dense)
        k = int(self.config["rrf_k"])
        fused = {
            doc.document_id:
            sparse_weight / (k + sparse_rank[doc.document_id])
            + dense_weight / (k + dense_rank[doc.document_id])
            for doc in candidates
        }
        return sorted(candidates, key=lambda doc: (-fused[doc.document_id], doc.document_id))

    def _encode_documents(self) -> dict[str, list[float]]:
        if self._embedding_backend == "hash_test_only":
            return {
                doc.document_id: _semantic_embedding(doc.text)
                for doc in self.documents
            }
        if self._embedding_backend != "sentence_transformers_local":
            raise ValueError(
                f"Unsupported dense embedding backend: {self._embedding_backend}"
            )
        model_path = (
            self.knowledge_root.parent
            / str(self.config["embedding"]["local_model_path"])
        ).resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(
                "The frozen dense embedding model is missing. Run "
                "experiments/prepare_embedding_model.py before using RAG: "
                f"{model_path}"
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the formal dense retriever"
            ) from exc
        self._embedding_model = SentenceTransformer(
            str(model_path),
            device="cpu",
            local_files_only=True,
        )
        matrix = self._embedding_model.encode(
            [doc.text for doc in self.documents],
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if matrix.shape != (
            len(self.documents),
            int(self.config["embedding"]["dimension"]),
        ):
            raise ValueError(f"Unexpected dense embedding matrix shape: {matrix.shape}")
        return {
            doc.document_id: np.asarray(matrix[index], dtype=float).tolist()
            for index, doc in enumerate(self.documents)
        }

    def _encode_query(self, query: str) -> list[float]:
        if self._embedding_backend == "hash_test_only":
            return _semantic_embedding(query)
        if self._embedding_model is None:
            raise RuntimeError("Dense embedding model was not initialized")
        vector = self._embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vector, dtype=float).tolist()

    @staticmethod
    def _gate(
        candidates: list[KnowledgeDocument], metadata: dict[str, set[str]]
    ) -> list[KnowledgeDocument]:
        fallbacks = {
            "business_types": {"general_coordination"},
            "time_scales": {"general"},
            "spatial_levels": {"general"},
        }
        gated: list[KnowledgeDocument] = []
        for doc in candidates:
            if all(
                not requested
                or bool(getattr(doc, field) & requested)
                or bool(getattr(doc, field) & fallbacks.get(field, set()))
                for field, requested in metadata.items()
            ):
                gated.append(doc)
        return gated

    @staticmethod
    def _as_evidence(doc: KnowledgeDocument) -> EvidenceRef:
        return EvidenceRef(
            evidence_id=doc.document_id,
            source=doc.source,
            locator=doc.locator,
            sha256=doc.sha256,
            excerpt=" ".join(doc.text.split())[:280],
        )

    def _load_config(self) -> dict[str, object]:
        payload = json.loads((self.knowledge_root / "retrieval_config.json").read_text(encoding="utf-8"))
        if payload["method_id"] != self.method_id:
            raise ValueError("Retrieval config method_id does not match the implementation")
        return payload

    def _load_documents(self) -> list[KnowledgeDocument]:
        documents: list[KnowledgeDocument] = []
        for path in sorted(self.knowledge_root.rglob("records.jsonl")):
            domain = path.relative_to(self.knowledge_root).parts[0]
            if domain not in {"building", "grid", "cross_domain"}:
                continue
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("domain") != domain:
                    raise ValueError(f"Knowledge domain mismatch at {path}:{line_number}")
                source_ids = sorted({str(ref["source_id"]) for ref in payload["source_refs"]})
                indexed_fields = (
                    "title", "summary", "keywords", "entities", "relations", "rules",
                    "applicability", "limitations", "units", "time_scales", "spatial_levels",
                    "business_types", "directions",
                )
                text_parts: list[str] = []
                for field in indexed_fields:
                    value = payload.get(field, "")
                    text_parts.append(" ".join(map(str, value)) if isinstance(value, list) else str(value))
                canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                document_id = str(payload["record_id"])
                documents.append(
                    KnowledgeDocument(
                        document_id=document_id,
                        domain=domain,
                        text="\n".join(part for part in text_parts if part),
                        source=f"knowledge:{domain}:{'+'.join(source_ids)}",
                        locator=(str(path.relative_to(self.knowledge_root.parent)).replace("\\", "/") + f"#{document_id}"),
                        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                        record_type=str(payload["record_type"]),
                        units=frozenset(map(str, payload["units"])),
                        time_scales=frozenset(map(str, payload["time_scales"])),
                        spatial_levels=frozenset(map(str, payload["spatial_levels"])),
                        applicable_roles=frozenset(map(str, payload["applicable_roles"])),
                        business_types=frozenset(map(str, payload["business_types"])),
                        directions=frozenset(map(str, payload["directions"])),
                    )
                )
        ids = [doc.document_id for doc in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Knowledge document IDs must be globally unique")
        return documents


def _normalize_direction(direction: str) -> str:
    aliases = {
        "B2G": "building_to_grid",
        "G2B": "grid_to_building",
        "building_energy_manager->distribution_grid_operator": "building_to_grid",
        "distribution_grid_operator->building_energy_manager": "grid_to_building",
    }
    normalized = aliases.get(direction, direction)
    if normalized not in {"building_to_grid", "grid_to_building"}:
        raise ValueError(f"Unsupported transfer direction: {direction}")
    return normalized


def _rank(scores: dict[str, float]) -> dict[str, int]:
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return {key: index + 1 for index, key in enumerate(ordered)}
