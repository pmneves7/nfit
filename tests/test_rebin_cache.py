"""The compressed tier remains bounded and never writes runtime spill files."""

import gc
import weakref
import zipfile
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread

import numpy as np

from nfit.analysis.artifacts import read_dataset_artifact
from nfit.cache_utils import array_payload_nbytes, lru_store
from nfit.rebin_cache import (
    CompressedBinning,
    RebinCache,
    RebinCacheBudget,
    _DiskBinning,
)
from tests.project_gui_test_support import _tiny_mdhisto_data


def test_compressed_binning_roundtrip_and_explicit_npz_save(tmp_path):
    source = _tiny_mdhisto_data(3.0)
    artifact = CompressedBinning.from_data(source, max_bytes=10_000)
    assert artifact is not None
    restored = artifact.restore()
    np.testing.assert_allclose(restored.signal, source.signal, equal_nan=True)
    np.testing.assert_allclose(restored.errors, source.errors, equal_nan=True)
    assert not restored.signal.flags.writeable
    assert not restored.errors.flags.writeable

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


def test_unreadable_lazy_project_binning_falls_back_to_cache_miss(tmp_path):
    member = "assets/binnings/cache/data.npz"
    for filename, include_member in (("missing.nfit", False), ("bad.nfit", True)):
        project = tmp_path / filename
        with zipfile.ZipFile(project, "w") as archive:
            archive.writestr("project.json", b"{}")
            if include_member:
                archive.writestr(member, b"not an npz artifact")
        cache = RebinCache()
        cache.set_project_backing(
            "key",
            signature="signature",
            project_path=project,
            member=member,
            lazy=True,
        )

        sentinel = object()
        assert cache.get("key", sentinel) is sentinel
        assert "key" not in cache


def test_viewer_and_composite_stages_share_one_global_budget():
    budget = RebinCacheBudget()
    viewer = RebinCache(budget)
    composite = RebinCache(budget)
    total_limit = 80

    lru_store(viewer, "viewer", ("v", _tiny_mdhisto_data(2.0)), None, total_limit)
    lru_store(
        composite, "composite", ("c", _tiny_mdhisto_data(3.0)), None, total_limit
    )

    assert budget.total_bytes() <= total_limit
    assert "viewer" not in viewer
    assert composite.has_signature("composite", "c")


def test_live_compressed_result_is_reused_and_counted_until_released():
    budget = RebinCacheBudget()
    cache = RebinCache(budget)
    source = _tiny_mdhisto_data(4.0)
    artifact = CompressedBinning.from_data(source, max_bytes=10_000)
    assert artifact is not None
    cache._compressed["key"] = ("signature", artifact)
    cache._compressed_bytes = artifact.nbytes
    cache._compressed_ticks["key"] = budget.next_tick()

    first = cache.get("key")[1]
    borrowed_bytes = budget.total_bytes() - artifact.nbytes
    assert borrowed_bytes > 0
    second = cache.get("key")[1]
    assert second is first
    assert second.signal is first.signal

    reference = weakref.ref(first)
    del first, second
    gc.collect()
    assert reference() is None
    assert budget.total_bytes() == artifact.nbytes


def test_concurrent_compressed_lookup_decodes_once(monkeypatch):
    budget = RebinCacheBudget()
    cache = RebinCache(budget)
    artifact = CompressedBinning.from_data(
        _tiny_mdhisto_data(5.0), max_bytes=10_000
    )
    assert artifact is not None
    cache._compressed["key"] = ("signature", artifact)
    cache._compressed_bytes = artifact.nbytes
    cache._compressed_ticks["key"] = budget.next_tick()
    calls = 0
    original = artifact.restore

    def restore():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(artifact, "restore", restore)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _index: cache.get("key")[1], range(4)))

    assert calls == 1
    assert all(result is results[0] for result in results)


def test_replacing_signature_never_returns_stale_borrowed_result():
    budget = RebinCacheBudget()
    cache = RebinCache(budget)
    old = CompressedBinning.from_data(_tiny_mdhisto_data(1.0), max_bytes=10_000)
    assert old is not None
    cache._compressed["key"] = ("old", old)
    cache._compressed_bytes = old.nbytes
    cache._compressed_ticks["key"] = budget.next_tick()
    old_result = cache.get("key")[1]

    cache["key"] = ("new", _tiny_mdhisto_data(9.0))
    signature, new_result = cache.get("key")
    assert signature == "new"
    assert new_result is not old_result
    np.testing.assert_allclose(new_result.signal, 9.0)


def test_live_borrow_is_an_irreducible_budget_floor_without_eviction_loop():
    budget = RebinCacheBudget()
    cache = RebinCache(budget)
    artifact = CompressedBinning.from_data(
        _tiny_mdhisto_data(6.0), max_bytes=10_000
    )
    assert artifact is not None
    cache._compressed["key"] = ("signature", artifact)
    cache._compressed_bytes = artifact.nbytes
    cache._compressed_ticks["key"] = budget.next_tick()
    result = cache.get("key")[1]

    budget.configure(1)
    budget.enforce()

    assert budget.total_bytes() > 1
    assert "key" in cache
    assert cache.get("key")[1] is result
    assert "key" not in cache._compressed


def test_live_resident_result_remains_canonical_after_budget_eviction():
    budget = RebinCacheBudget()
    cache = RebinCache(budget)
    result = _tiny_mdhisto_data(7.0)
    cache["key"] = ("signature", result)
    resident_bytes = budget.total_bytes()
    assert resident_bytes == array_payload_nbytes((result,))

    budget.configure(max(resident_bytes // 2, 1))
    budget.enforce()

    assert cache.peek_resident("key") is None
    assert cache.get("key")[1] is result
    assert budget.total_bytes() == resident_bytes


def test_disk_restore_remains_canonical_when_budget_immediately_evicts_it(tmp_path):
    budget = RebinCacheBudget()
    cache = RebinCache(budget)
    source = _tiny_mdhisto_data(8.0)
    artifact = CompressedBinning.from_data(source, max_bytes=10_000)
    assert artifact is not None
    path = tmp_path / "disk-result.npz"
    artifact.write_npz(path)
    cache._disk["key"] = _DiskBinning("signature", path, owned=False)
    budget.configure(1)

    first = cache.get("key")[1]
    assert cache.peek_resident("key") is None
    second = cache.get("key")[1]

    assert second is first
    assert second.signal is first.signal


def test_replacement_cannot_race_with_inflight_restore(monkeypatch):
    budget = RebinCacheBudget()
    cache = RebinCache(budget)
    artifact = CompressedBinning.from_data(
        _tiny_mdhisto_data(2.0), max_bytes=10_000
    )
    assert artifact is not None
    cache._compressed["key"] = ("old", artifact)
    cache._compressed_bytes = artifact.nbytes
    cache._compressed_ticks["key"] = budget.next_tick()
    restoring = Event()
    resume = Event()
    original = artifact.restore

    def restore():
        restoring.set()
        assert resume.wait(timeout=5)
        return original()

    monkeypatch.setattr(artifact, "restore", restore)
    lookup = Thread(target=cache.get, args=("key",))
    lookup.start()
    assert restoring.wait(timeout=5)
    replacement = Thread(
        target=cache.__setitem__,
        args=("key", ("new", _tiny_mdhisto_data(9.0))),
    )
    replacement.start()
    resume.set()
    lookup.join(timeout=5)
    replacement.join(timeout=5)

    assert not lookup.is_alive()
    assert not replacement.is_alive()
    signature, result = cache.get("key")
    assert signature == "new"
    np.testing.assert_allclose(result.signal, 9.0)
