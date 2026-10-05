"""Allocation admission is global, nested, and released after failed jobs."""

from contextvars import copy_context
from threading import Event, Thread

import pytest

from nfit import resource_budget
from nfit.resource_budget import ResourceLimitError, check_memory, reserve_memory, snapshot_memory


@pytest.fixture(autouse=True)
def controlled_memory(monkeypatch):
    monkeypatch.setattr(resource_budget, "_rss_provider", lambda: 100)
    monkeypatch.setattr(resource_budget, "_limit_provider", lambda: 1000)
    monkeypatch.setattr(resource_budget, "_available_provider", lambda: 2000)
    monkeypatch.setattr(resource_budget, "_managed_provider", lambda: 0)
    assert snapshot_memory().reserved_bytes == 0
    yield
    assert snapshot_memory().reserved_bytes == 0


def test_rejected_reservation_has_actionable_fields_and_leaves_no_pending_charge():
    with pytest.raises(ResourceLimitError) as error:
        with reserve_memory(901, operation="Loading NiO"):
            pytest.fail("must reject before allocation")
    assert error.value.operation == "Loading NiO"
    assert error.value.requested_bytes == 901
    assert error.value.used_bytes == 100
    assert error.value.limit_bytes == 1000
    assert error.value.missing_bytes == 1
    assert snapshot_memory().reserved_bytes == 0
    with reserve_memory(900):
        assert snapshot_memory().remaining_bytes == 0


def test_nested_operations_reuse_peak_and_only_reserve_the_excess():
    with reserve_memory(400):
        with reserve_memory(250):
            assert snapshot_memory().reserved_bytes == 400
        with reserve_memory(600):
            assert snapshot_memory().reserved_bytes == 600
            check_memory(600)
        assert snapshot_memory().reserved_bytes == 400
    assert snapshot_memory().reserved_bytes == 0


def test_reservations_release_after_cancellation_or_exception():
    with pytest.raises(RuntimeError):
        with reserve_memory(500):
            raise RuntimeError("cancelled")
    assert snapshot_memory().reserved_bytes == 0


def test_nested_scopes_closed_out_of_order_keep_only_live_reservations():
    outer = reserve_memory(400)
    middle = reserve_memory(700)
    suspended = reserve_memory(600)
    outer.__enter__()
    middle.__enter__()
    suspended.__enter__()
    try:
        assert snapshot_memory().reserved_bytes == 700
        middle.__exit__(None, None, None)
        assert snapshot_memory().reserved_bytes == 600
        outer.__exit__(None, None, None)
        assert snapshot_memory().reserved_bytes == 600
        # The suspended child remains charged even after the owner leaves.
        with pytest.raises(ResourceLimitError):
            with reserve_memory(301):
                pytest.fail("must still account for the live suspended scope")
    finally:
        suspended.__exit__(None, None, None)
    assert snapshot_memory().reserved_bytes == 0


def test_concurrent_jobs_cannot_spend_the_same_remaining_memory():
    entered = Event()
    release = Event()

    def first_job():
        with reserve_memory(800):
            entered.set()
            assert release.wait(5)

    worker = Thread(target=first_job)
    worker.start()
    assert entered.wait(5)
    try:
        with pytest.raises(ResourceLimitError):
            with reserve_memory(101):
                pytest.fail("second job must reject")
        with reserve_memory(100):
            assert snapshot_memory().reserved_bytes == 900
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()


def test_copied_context_threads_get_independent_reservations():
    errors = []
    with reserve_memory(800):
        inherited = copy_context()

        def child():
            with reserve_memory(100):
                assert snapshot_memory().reserved_bytes == 900
            try:
                with reserve_memory(101):
                    pytest.fail("must charge a copied context's worker separately")
            except ResourceLimitError as error:
                errors.append(error)

        worker = Thread(target=lambda: inherited.run(child))
        worker.start()
        worker.join(5)
        assert not worker.is_alive()
        assert snapshot_memory().reserved_bytes == 800
    assert len(errors) == 1


def test_expired_copied_context_cannot_reuse_a_released_reservation():
    with reserve_memory(800):
        copied = copy_context()
    with pytest.raises(ResourceLimitError):
        copied.run(check_memory, 901)


def test_available_system_memory_is_checked_independently(monkeypatch):
    monkeypatch.setattr(resource_budget, "_available_provider", lambda: 50)
    with pytest.raises(ResourceLimitError) as error:
        check_memory(51)
    assert error.value.available_bytes == 50
    assert error.value.limit_bytes == 150
    assert error.value.missing_bytes == 1


def test_untouched_managed_storage_is_charged_even_when_rss_is_smaller(monkeypatch):
    monkeypatch.setattr(resource_budget, "_managed_provider", lambda: 800)
    assert snapshot_memory().used_bytes == 800
    with pytest.raises(ResourceLimitError):
        check_memory(201)


@pytest.mark.parametrize("reducer", ["raw", "mdevent", "powder"])
def test_dgs_admission_precedes_source_access(reducer):
    from nfit.mdevent import bin_mdevent_group, bin_mdevent_powder_group
    from nfit.pipeline import DatasetGroup
    from nfit.raw_dgs import bin_raw_dgs_group

    functions = {"raw": bin_raw_dgs_group, "mdevent": bin_mdevent_group, "powder": bin_mdevent_powder_group}
    dimensions = 2 if reducer == "powder" else 4
    # No source/config exists: an admitted call would reach a KeyError. The
    # peak guard must reject before any scientific file or workspace is read.
    with pytest.raises(ResourceLimitError):
        functions[reducer](DatasetGroup("test"), lower=[0] * dimensions,
                           upper=[1] * dimensions, num_bins=[10] * dimensions)


@pytest.mark.parametrize("stream", [False, True])
def test_resolved_rebin_grid_rejects_before_output_allocation(monkeypatch, stream):
    from nfit import ArrayRebinSource, NDRebin, rebin_nd, rebin_nd_stream

    monkeypatch.setattr(NDRebin, "_empty_accumulators", lambda *args: pytest.fail("output allocation"))
    kwargs = dict(lower=[0, 0], upper=[1, 1], num_bins=[10, 10], backend="numpy")
    with pytest.raises(ResourceLimitError):
        if stream:
            rebin_nd_stream(ArrayRebinSource([1.], [[0., 0.]]), **kwargs)
        else:
            rebin_nd([1.], [[0., 0.]], **kwargs)
