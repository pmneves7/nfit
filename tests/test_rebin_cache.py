"""The compressed tier remains bounded and never writes runtime spill files."""

import numpy as np

from nfit.analysis.artifacts import read_dataset_artifact
from nfit.cache_utils import lru_store
from nfit.rebin_cache import CompressedBinning, RebinCache, RebinCacheBudget
from tests.project_gui_test_support import _tiny_mdhisto_data


def test_compressed_binning_roundtrip_and_explicit_npz_save(tmp_path):
    source = _tiny_mdhisto_data(3.0)
    artifact = CompressedBinning.from_data(source, max_bytes=10_000)
    assert artifact is not None
    restored = artifact.restore()
    np.testing.assert_allclose(restored.signal, source.signal, equal_nan=True)
    np.testing.assert_allclose(restored.errors, source.errors, equal_nan=True)

    path = tmp_path / "chosen-result.npz"
    artifact.write_npz(path)
    assert path.read_bytes()[:2] == b"PK"
    from_disk = read_dataset_artifact(path)
    np.testing.assert_allclose(from_disk.signal, source.signal, equal_nan=True)


def test_oldest_compressed_binning_is_offered_before_eviction(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(
        tempfile,
        "TemporaryDirectory",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("disk spill")),
    )
    cache = RebinCache()
    offered = []

    def save_before_discard(key, artifact):
        offered.append(key)
        artifact.write_npz(tmp_path / f"{key}.npz")

    cache.before_discard = save_before_discard
    artifact = CompressedBinning.from_data(
        _tiny_mdhisto_data(1.0), max_bytes=10_000
    )
    assert artifact is not None
    budget = artifact.nbytes + artifact.nbytes // 2
    for key in ("first", "second"):
        cache.set_label(key, f"named {key}")
        lru_store(cache, key, (key, _tiny_mdhisto_data(1.0)), 0, budget)

    assert offered == ["named first"]
    assert "first" not in cache
    assert cache.has_signature("second", "second")
    assert cache._compressed_bytes <= cache._compressed_limit
    np.testing.assert_allclose(
        read_dataset_artifact(tmp_path / "named first.npz").signal,
        _tiny_mdhisto_data(1.0).signal,
    )


def test_evicted_binning_uses_and_cleans_session_disk_cache(tmp_path):
    cache = RebinCache()
    destination = tmp_path / "session-first.npz"

    def spill_to_disk(_key, artifact):
        artifact.write_npz(destination)
        return destination

    cache.before_discard = spill_to_disk
    artifact = CompressedBinning.from_data(
        _tiny_mdhisto_data(1.0), max_bytes=10_000
    )
    assert artifact is not None
    budget = artifact.nbytes + artifact.nbytes // 2
    for key in ("first", "second"):
        lru_store(cache, key, (key, _tiny_mdhisto_data(1.0)), 0, budget)

    assert destination.exists()
    assert cache.has_signature("first", "first")
    signature, restored = cache.get("first")
    assert signature == "first"
    np.testing.assert_allclose(restored.signal, _tiny_mdhisto_data(1.0).signal)

    cache.clear()
    assert not destination.exists()


def test_viewer_and_composite_stages_share_one_global_budget():
    budget = RebinCacheBudget()
    viewer = RebinCache(budget)
    composite = RebinCache(budget)
    viewer_data = _tiny_mdhisto_data(2.0)
    composite_data = _tiny_mdhisto_data(3.0)
    total_limit = 80

    lru_store(viewer, "viewer", ("v", viewer_data), None, total_limit)
    lru_store(composite, "composite", ("c", composite_data), None, total_limit)

    assert budget.total_bytes() <= total_limit
    assert "viewer" not in viewer
    assert composite.has_signature("composite", "c")
