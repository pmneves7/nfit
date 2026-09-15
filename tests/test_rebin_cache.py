"""The compressed tier remains bounded and never writes runtime spill files."""

import numpy as np

from nfit.analysis.artifacts import read_dataset_artifact
from nfit.cache_utils import lru_store
from nfit.rebin_cache import CompressedBinning, RebinCache
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
    for key in ("first", "second"):
        cache.set_label(key, f"named {key}")
        lru_store(cache, key, (key, _tiny_mdhisto_data(1.0)), 0, 2_000)

    assert offered == ["named first"]
    assert "first" not in cache
    assert cache.has_signature("second", "second")
    assert cache._compressed_bytes <= cache._compressed_limit
    np.testing.assert_allclose(
        read_dataset_artifact(tmp_path / "named first.npz").signal,
        _tiny_mdhisto_data(1.0).signal,
    )
