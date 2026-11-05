from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from b2g_agent.contracts import Episode, EpisodeGroundTruth, TaskFamily


CASE_ID = "ieee33_10bem_15min"


def find_repo_root(start: Path | None = None) -> Path:
    candidate = (start or Path(__file__)).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for path in (candidate, *candidate.parents):
        if (path / "pyproject.toml").is_file() and (path / "src" / "b2g_agent").is_dir():
            return path
    raise FileNotFoundError("Could not locate the B2G-Agent repository root")


@dataclass(frozen=True)
class CaseReadiness:
    ready_for_contract_evaluation: bool
    ready_for_physics: bool
    missing_local_assets: tuple[str, ...]
    local_data_root: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready_for_contract_evaluation": self.ready_for_contract_evaluation,
            "ready_for_physics": self.ready_for_physics,
            "missing_local_assets": list(self.missing_local_assets),
            "local_data_root": str(self.local_data_root),
        }


class CaseBundle:
    """Loads one immutable formal case without importing agent implementation code."""

    def __init__(self, case_dir: Path) -> None:
        self.case_dir = case_dir.resolve()
        self.case_file = self.case_dir / "case.json"
        if not self.case_file.is_file():
            raise FileNotFoundError(f"Formal case manifest is missing: {self.case_file}")
        self.manifest = json.loads(self.case_file.read_text(encoding="utf-8"))
        if self.manifest.get("case_id") != CASE_ID:
            raise ValueError(f"Unsupported case_id: {self.manifest.get('case_id')}")
        if self.manifest.get("time_resolution_minutes") != 15:
            raise ValueError("The formal case must use 15-minute resolution")

    @classmethod
    def default(cls, repo_root: Path | None = None) -> CaseBundle:
        root = repo_root or find_repo_root()
        return cls(root / "cases" / CASE_ID)

    @property
    def case_id(self) -> str:
        return str(self.manifest["case_id"])

    @property
    def episodes_path(self) -> Path:
        return self.case_dir / "tasks" / "episodes.jsonl"

    @property
    def groundtruth_path(self) -> Path:
        return self.case_dir / "groundtruth" / "contract_oracles.jsonl"

    @property
    def local_data_root(self) -> Path:
        configured = os.getenv("B2G_DATA_ROOT", "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return find_repo_root(self.case_dir) / ".local-data"

    def list_episodes(
        self,
        *,
        family: TaskFamily | str | None = None,
        split: str | None = None,
        limit: int | None = None,
    ) -> list[Episode]:
        selected: list[Episode] = []
        normalized_family = TaskFamily(family) if family else None
        for episode in self.iter_episodes():
            if normalized_family and normalized_family not in episode.task_families:
                continue
            if split and episode.split.value != split:
                continue
            selected.append(episode)
            if limit is not None and len(selected) >= limit:
                break
        return selected

    def iter_episodes(self) -> Iterable[Episode]:
        if not self.episodes_path.is_file():
            raise FileNotFoundError(f"Episode corpus is missing: {self.episodes_path}")
        with self.episodes_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    yield Episode.model_validate_json(line)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid episode at {self.episodes_path}:{line_number}: {exc}"
                    ) from exc

    def get_episode(self, episode_id: str) -> Episode:
        for episode in self.iter_episodes():
            if episode.episode_id == episode_id:
                return episode
        raise KeyError(f"Unknown episode: {episode_id}")

    def iter_groundtruth(self) -> Iterable[EpisodeGroundTruth]:
        """Load private evaluation oracles; web handlers must never serialize these records."""
        if not self.groundtruth_path.is_file():
            raise FileNotFoundError(
                f"Ground-truth corpus is missing: {self.groundtruth_path}"
            )
        with self.groundtruth_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    yield EpisodeGroundTruth.model_validate_json(line)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid ground truth at {self.groundtruth_path}:{line_number}: {exc}"
                    ) from exc

    def get_groundtruth(self, episode_id: str) -> EpisodeGroundTruth:
        for truth in self.iter_groundtruth():
            if truth.episode_id == episode_id:
                return truth
        raise KeyError(f"Unknown ground truth: {episode_id}")

    def validate_groundtruth_alignment(self) -> tuple[str, ...]:
        """Check one-to-one linkage without treating unverified physics as verified truth."""
        if not self.episodes_path.is_file():
            return ("missing_episode_corpus",)
        if not self.groundtruth_path.is_file():
            return ("missing_groundtruth_corpus",)
        episodes = list(self.iter_episodes())
        truths = list(self.iter_groundtruth())
        errors: list[str] = []
        episode_by_id = {episode.episode_id: episode for episode in episodes}
        truth_by_id = {truth.episode_id: truth for truth in truths}
        if len(episode_by_id) != len(episodes):
            errors.append("duplicate_episode_id")
        if len(truth_by_id) != len(truths):
            errors.append("duplicate_groundtruth_episode_id")
        if set(episode_by_id) != set(truth_by_id):
            errors.append("episode_groundtruth_id_mismatch")
        for episode_id in sorted(set(episode_by_id) & set(truth_by_id)):
            episode = episode_by_id[episode_id]
            truth = truth_by_id[episode_id]
            source_hash = hashlib.sha256(
                episode.sender_message.encode("utf-8")
            ).hexdigest()
            if truth.case_id != episode.case_id:
                errors.append(f"case_id_mismatch:{episode_id}")
            if truth.source_message_sha256 != source_hash:
                errors.append(f"source_hash_mismatch:{episode_id}")
            if truth.task_specification.episode_id != episode_id:
                errors.append(f"task_specification_id_mismatch:{episode_id}")
            if truth.task_specification.source_text != episode.sender_message:
                errors.append(f"task_specification_source_mismatch:{episode_id}")
        return tuple(errors)

    def bem_mapping(self) -> dict[str, int]:
        path = self.case_dir / "network" / "bem_to_bus.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle)
            mapping = {row["bem_id"]: int(row["bus"] or 0) for row in rows}
        if len(mapping) != 10:
            raise ValueError("The formal case must map exactly 10 BEMs")
        return mapping

    def readiness(self) -> CaseReadiness:
        local_case = self.local_data_root / self.case_id
        required_local = [
            local_case / "weather" / "USA_NY_Buffalo.Niagara.Intl_.AP_.725280_TMY3.epw",
            *[
                local_case / "buildings" / f"bem_{index:02d}" / "energyplus_23_1.idf"
                for index in range(1, 11)
            ],
            local_case / "profiles" / "background_load_15min.csv",
            local_case / "profiles" / "background_load_manifest.json",
            local_case / "profiles" / "bem_baseline_15min.csv",
            local_case / "profiles" / "bem_baseline_manifest.json",
            local_case / "profiles" / "pv_15min.csv",
            local_case / "profiles" / "pv_manifest.json",
            local_case / "profiles" / "prices_15min.csv",
            local_case / "profiles" / "reserve_15min.csv",
            local_case / "profiles" / "market_manifest.json",
            local_case / "profiles" / "daily_metrics.csv",
            local_case / "profiles" / "daily_metrics_manifest.json",
        ]
        required_case = [
            self.case_dir / "network" / "line_limits.csv",
            self.case_dir / "network" / "transformer.json",
            self.case_dir / "days" / "target_days.json",
        ]
        missing_items = [
            str(path.relative_to(self.local_data_root))
            for path in required_local
            if not path.is_file()
        ] + [
            f"case:{path.relative_to(self.case_dir).as_posix()}"
            for path in required_case
            if not path.is_file()
        ]
        smoke_report_path = local_case / "energyplus_smoke_report.json"
        if not smoke_report_path.is_file():
            missing_items.append(
                f"{self.case_id}{os.sep}energyplus_smoke_report.json"
            )
        else:
            smoke_report = json.loads(smoke_report_path.read_text(encoding="utf-8"))
            if smoke_report.get("status") != "PASS":
                blocked = ",".join(smoke_report.get("blocked_bems", []))
                missing_items.append(f"local:energyplus_smoke_report:FAIL[{blocked}]")
        missing = tuple(missing_items)
        contract_ready = (
            self.episodes_path.is_file()
            and self.groundtruth_path.is_file()
            and not self.validate_groundtruth_alignment()
        )
        return CaseReadiness(
            ready_for_contract_evaluation=contract_ready,
            ready_for_physics=not missing,
            missing_local_assets=missing,
            local_data_root=self.local_data_root,
        )

    def public_summary(self) -> dict[str, Any]:
        episodes = list(self.iter_episodes())
        truths = list(self.iter_groundtruth()) if self.groundtruth_path.is_file() else []
        family_counts: dict[str, int] = {}
        split_counts: dict[str, int] = {}
        for episode in episodes:
            family_counts[episode.primary_family.value] = (
                family_counts.get(episode.primary_family.value, 0) + 1
            )
            split_counts[episode.split.value] = split_counts.get(episode.split.value, 0) + 1
        return {
            **self.manifest,
            "episode_count": len(episodes),
            "episode_sha256": sha256_file(self.episodes_path),
            "groundtruth_count": len(truths),
            "groundtruth_sha256": (
                sha256_file(self.groundtruth_path)
                if self.groundtruth_path.is_file()
                else None
            ),
            "groundtruth_contract_status_counts": _count_values(
                truth.contract_truth_status for truth in truths
            ),
            "groundtruth_physics_status_counts": _count_values(
                truth.physics_truth_status for truth in truths
            ),
            "groundtruth_alignment_errors": list(
                self.validate_groundtruth_alignment()
            ),
            "family_counts": family_counts,
            "split_counts": split_counts,
            "readiness": self.readiness().as_dict(),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
