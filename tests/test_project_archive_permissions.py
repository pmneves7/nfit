"""Atomic saves preserve intentional sharing permissions and save timestamps."""

import os
import stat

import pytest

from nfit.project_archive import read_project_manifest, write_project_manifest


@pytest.mark.skipif(os.name == "nt", reason="POSIX file-mode sharing contract")
def test_atomic_save_retains_shared_destination_mode_and_updates_timestamp(tmp_path):
    path = tmp_path / "shared.nfit"
    write_project_manifest(path, {"value": "before"})
    path.chmod(0o660)
    old_time = 1_600_000_000_000_000_000
    os.utime(path, ns=(old_time, old_time))
    write_project_manifest(path, {"value": "after"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o660
    assert path.stat().st_mtime_ns > old_time
    assert read_project_manifest(path)["value"] == "after"


@pytest.mark.skipif(not hasattr(os, "setxattr"), reason="No extended file attributes")
def test_atomic_save_retains_supported_file_access_metadata(tmp_path):
    path = tmp_path / "shared.nfit"
    write_project_manifest(path, {"value": "before"})
    key = "user.nfit-test-sharing"
    try:
        os.setxattr(path, key, b"shared-access-metadata")
    except OSError:
        pytest.skip("Filesystem does not support this test extended attribute")
    write_project_manifest(path, {"value": "after"})
    assert os.getxattr(path, key) == b"shared-access-metadata"
