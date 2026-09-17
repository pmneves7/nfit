from collections import OrderedDict

import numpy as np

import nfit.performance as performance
from nfit.cache_utils import (
    array_payload_nbytes,
    lru_store,
    scientific_cache_budget_bytes,
)


def test_scientific_cache_budget_uses_configured_rebin_total(monkeypatch):
    monkeypatch.setattr(
        performance,
        "available_memory_bytes",
        lambda: 16 * 1024**3,
    )
    monkeypatch.setattr(
        performance,
        "load_performance_settings",
        lambda: {
            "max_batch_mb": 0,
            "workers": 0,
            "transient_memory_percent": 40,
        },
    )

    assert scientific_cache_budget_bytes() == int(16 * 1024**3 * 0.4)


def test_array_payload_nbytes_counts_shared_arrays_once():
    first = np.zeros(4, dtype=np.float64)
    second = np.zeros(3, dtype=np.float32)

    assert array_payload_nbytes({"a": first, "again": [first], "b": second}) == (
        first.nbytes + second.nbytes
    )


def test_lru_store_evicts_to_array_byte_budget():
    cache = OrderedDict()
    first = np.zeros(2, dtype=np.float64)
    second = np.zeros(2, dtype=np.float64)

    lru_store(cache, "first", first, limit=4, max_array_bytes=first.nbytes)
    lru_store(cache, "second", second, limit=4, max_array_bytes=second.nbytes)

    assert list(cache) == ["second"]


def test_lru_store_does_not_retain_oversized_entry():
    cache = OrderedDict()

    lru_store(
        cache,
        "large",
        np.zeros(4, dtype=np.float64),
        limit=4,
        max_array_bytes=8,
    )

    assert not cache
