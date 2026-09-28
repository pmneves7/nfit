from types import SimpleNamespace

from nfit.resource_usage import ResourceUsageSampler


def test_sampler_primes_cpu_and_normalizes_sample(monkeypatch):
    calls = []

    class FakeProcess:
        def cpu_percent(self, *, interval):
            calls.append(("process_cpu", interval))
            return 160.0

        def memory_info(self):
            return SimpleNamespace(rss=2_000)

    fake_psutil = SimpleNamespace(
        Process=lambda: FakeProcess(),
        cpu_count=lambda *, logical: 4,
        cpu_percent=lambda *, interval: calls.append(("system_cpu", interval)) or 37.5,
        virtual_memory=lambda: SimpleNamespace(total=10_000, percent=62.5),
    )
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)

    sampler = ResourceUsageSampler()
    snapshot = sampler.sample()

    assert calls == [
        ("process_cpu", None),
        ("system_cpu", None),
        ("process_cpu", None),
        ("system_cpu", None),
    ]
    assert snapshot.process_cpu_percent == 40.0
    assert snapshot.process_memory_percent == 20.0
    assert snapshot.system_cpu_percent == 37.5
    assert snapshot.system_memory_percent == 62.5


def test_sampler_clamps_process_cpu_to_single_machine_scale(monkeypatch):
    class FakeProcess:
        def cpu_percent(self, *, interval):
            return 800.0

        def memory_info(self):
            return SimpleNamespace(rss=0)

    fake_psutil = SimpleNamespace(
        Process=lambda: FakeProcess(),
        cpu_count=lambda *, logical: 8,
        cpu_percent=lambda *, interval: 150.0,
        virtual_memory=lambda: SimpleNamespace(total=100, percent=120.0),
    )
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)

    snapshot = ResourceUsageSampler().sample()

    assert snapshot.process_cpu_percent == 100.0
    assert snapshot.process_memory_percent == 0.0
    assert snapshot.system_cpu_percent == 100.0
    assert snapshot.system_memory_percent == 100.0
