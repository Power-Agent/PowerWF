from pathlib import Path

from b2g_agent.observability.reproducibility import (
    PHYSICS_IMPLEMENTATION_PATHS,
    physics_implementation_bundle_sha256,
    physics_implementation_hashes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_physics_implementation_fingerprint_is_complete_and_stable() -> None:
    hashes = physics_implementation_hashes(REPO_ROOT)

    assert tuple(hashes) == PHYSICS_IMPLEMENTATION_PATHS
    assert all(len(digest) == 64 for digest in hashes.values())
    assert physics_implementation_bundle_sha256(REPO_ROOT) == (
        physics_implementation_bundle_sha256(REPO_ROOT)
    )

