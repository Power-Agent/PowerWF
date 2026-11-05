from pathlib import Path

import pytest

from b2g_agent.artifact_paths import resolve_recorded_path


@pytest.mark.parametrize(("recorded", "expected"), [
    ("outputs/benchmark/20261030T223320Z/w1/run/events.jsonl",
     "outputs/benchmark/proposed/w1/run/events.jsonl"),
    ("outputs/benchmark/a-v2/remaining150/w2/index.jsonl",
     "outputs/benchmark/ablation-actor/remaining150/w2/index.jsonl"),
    ("outputs/benchmark/20261102T150426Z-v4-r/lr/response.json",
     "outputs/benchmark/ablation-reviewer/first50/lr/response.json"),
    ("outputs/pc/owner/runs/evidence.json", "outputs/physics/cache/owner/runs/evidence.json"),
    ("outputs/pb/20261031T075835Z/runs/evidence.json",
     "outputs/physics/baseline/runs/evidence.json"),
    ("outputs/comparison/final/metrics.csv", "outputs/comparison/final/metrics.csv"),
    ("outputs/pc-other/record.json", "outputs/pc-other/record.json"),
])
def test_historical_paths_resolve_in_a_different_checkout(tmp_path, recorded, expected):
    assert resolve_recorded_path(recorded, repo_root=tmp_path) == tmp_path / expected
    absolute = "C:\\former-checkout\\B2G-Agent\\" + recorded.replace("/", "\\")
    assert resolve_recorded_path(absolute, repo_root=tmp_path) == tmp_path / expected


def test_existing_private_files_are_not_redirected(tmp_path):
    private = tmp_path / "outputs" / "pc" / "private.json"
    private.parent.mkdir(parents=True)
    private.write_text("{}", encoding="utf-8")
    assert resolve_recorded_path(private) == private
    assert resolve_recorded_path("unrelated.json", repo_root=tmp_path) == Path("unrelated.json")
