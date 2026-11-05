import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

from b2g_agent.publication import recorded_hash_matches
from b2g_agent.artifact_paths import resolve_recorded_path

ROOT = Path(__file__).resolve().parents[1]


def test_repository_prefixed_evidence_resolves_after_parent_join(tmp_path):
    reference = "B2G-Agent/outputs/physics/cache/run/evidence.json"
    expected = tmp_path / "outputs/physics/cache/run/evidence.json"
    parent = tmp_path / "outputs/benchmark/proposed"
    assert resolve_recorded_path(reference, repo_root=tmp_path) == expected
    assert resolve_recorded_path(parent / reference, repo_root=tmp_path) == expected
    assert resolve_recorded_path("B2G-Agent/.local-data/asset.json", repo_root=tmp_path) == tmp_path / ".local-data/asset.json"


def test_redaction_mapping_does_not_accept_unrecorded_changes(tmp_path):
    path = tmp_path / "outputs" / "journal.jsonl"
    path.parent.mkdir()
    path.write_bytes(b"published bytes")
    original = hashlib.sha256(b"acquisition bytes").hexdigest()
    published = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = tmp_path / "publication"
    metadata.mkdir()
    (metadata / "redaction_map.json").write_text(json.dumps({"schema_version": "1.0", "artifacts": {
        "outputs/journal.jsonl": {"original_sha256": original, "published_sha256": published}}}), encoding="utf-8")
    assert recorded_hash_matches(path, original, repo_root=tmp_path)
    assert recorded_hash_matches(path, published, repo_root=tmp_path)
    assert not recorded_hash_matches(path, "0" * 64, repo_root=tmp_path)
    path.write_bytes(b"unrecorded alteration")
    assert not recorded_hash_matches(path, original, repo_root=tmp_path)


def restore_module():
    spec = importlib.util.spec_from_file_location("restore_test", ROOT / "experiments/restore_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["../outside", "outputs/../../outside", "/outputs/file", "C:/outputs/file", "src/overwrite.py"])
def test_restore_rejects_escaping_and_unexpected_members(tmp_path, name):
    with pytest.raises(ValueError):
        restore_module().member_destination(tmp_path, zipfile.ZipInfo(name))


def test_restore_verifies_hashes_and_preserves_existing_files(tmp_path):
    downloads = tmp_path / "download"
    downloads.mkdir()
    archive = downloads / "b2g-input-data.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(".local-data/asset.txt", "release data")
    module = restore_module()
    manifest = {"schema_version": "1.0", "archives": [{"archive": archive.name, "sha256": module.sha256(archive)}]}
    (downloads / "release_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    assert module.restore(downloads, checkout) == 1
    assert module.restore(downloads, checkout) == 0
    (checkout / ".local-data/asset.txt").write_text("user content", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.restore(downloads, checkout)
    assert (checkout / ".local-data/asset.txt").read_text(encoding="utf-8") == "user content"


def test_restore_deeply_nested_physical_evidence(tmp_path):
    module = restore_module()
    downloads = tmp_path / "download"
    downloads.mkdir()
    member = "outputs/" + "/".join(["physical-evidence" * 4] * 4) + "/result.json"
    archive = downloads / "b2g-baseline.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(member, '{"converged":true}')
    manifest = {"schema_version": "1.0", "archives": [{"archive": archive.name, "sha256": module.sha256(archive)}]}
    (downloads / "release_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    assert module.restore(downloads, checkout, "all") == 1
    assert module.io_path(checkout / member).read_text(encoding="utf-8") == '{"converged":true}'
