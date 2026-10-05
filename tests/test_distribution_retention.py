"""Safety boundaries for destructive GitHub retention planning."""

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "distribution_retention", Path(__file__).resolve().parents[1] / "tools/distribution/retention.py")
retention = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retention)


def release(version, *, draft=False):
    names = []
    for platform in retention.PLATFORMS:
        suffix = '.tar.gz' if platform.startswith('linux') else '.exe' if platform.startswith('windows') else '.pkg'
        stem = f'nfit-{version}-{platform}'
        names.extend([stem + suffix, stem + '.json'])
    return {'tag_name': 'v' + version, 'draft': draft,
            'assets': [{'id': index, 'name': name} for index, name in enumerate(names)]}


def test_retains_newest_three_complete_semantic_versions_and_preserves_unknown_assets():
    old = release('0.9.0')
    old['assets'].append({'id': 999, 'name': 'scientific-data.zip'})
    items = [release('0.10.0'), old, release('0.11.0'), release('0.12.0')]
    plan = retention.plan_release_assets(items)
    assert len(plan) == 8
    assert {asset['tag'] for asset in plan} == {'v0.9.0'}
    assert 999 not in {asset['id'] for asset in plan}


def test_drafts_incomplete_unrecognized_tags_and_unuploaded_assets_are_preserved():
    incomplete = release('0.20.0')
    incomplete['assets'].pop()
    uploading = release('0.21.0')
    uploading['assets'][0]['state'] = 'new'
    nonversion = release('0.22.0')
    nonversion['tag_name'] = 'experiment'
    items = [release('0.30.0', draft=True), incomplete, uploading, nonversion,
             release('0.1.0'), release('0.2.0'), release('0.3.0')]
    assert retention.plan_release_assets(items) == []


def test_missing_checksum_or_linux_tarball_does_not_count_as_complete():
    item = release('0.1.0')
    item['assets'] = [asset for asset in item['assets'] if not asset['name'].endswith('.json')]
    assert not retention.complete(item, '0.1.0')
    item = release('0.1.0')
    item['assets'][0]['name'] = 'nfit-0.1.0-linux-x86_64.deb'
    assert not retention.complete(item, '0.1.0')


def test_artifacts_require_completed_run_and_age_strictly_exceeding_seven_days():
    now = datetime(2026, 10, 5, tzinfo=UTC)
    artifacts = [{'id': number, 'created_at': (now - timedelta(days=age)).isoformat(),
                  'workflow_run': {'id': number}}
                 for number, age in [(1, 8), (2, 8), (3, 7), (4, 8), (5, 8)]]
    statuses = {1: 'completed', 2: 'in_progress', 3: 'completed', 4: 'queued'}
    assert [item['id'] for item in retention.plan_artifacts(artifacts, statuses, now)] == [1]


def test_pagination_reads_all_pages(monkeypatch):
    seen = []

    def api(endpoint):
        seen.append(endpoint)
        return {'artifacts': list(range(100)) if endpoint.endswith('&page=1') else [100]}

    monkeypatch.setattr(retention, 'api', api)
    assert retention.pages('repos/owner/repo/actions/artifacts', 'artifacts') == list(range(101))
    assert len(seen) == 2


def test_invalid_retention_cannot_delete_everything():
    with pytest.raises(ValueError):
        retention.plan_release_assets([], keep=0)
    with pytest.raises(ValueError):
        retention.plan_artifacts([], {}, datetime.now(UTC), days=0)


def test_default_cli_only_reads_and_prints_plan(monkeypatch, capsys):
    items = [release(f'0.{index}.0') for index in range(1, 5)]
    for index, item in enumerate(items):
        item['id'] = index

    def pages(endpoint, key=None):
        if endpoint.endswith('/releases'):
            return items
        if endpoint.endswith('/assets'):
            return items[int(endpoint.split('/')[-2])]['assets']
        return []

    monkeypatch.setattr(retention, 'pages', pages)
    monkeypatch.setattr(retention, 'api', lambda *args, **kwargs: pytest.fail('Dry run called mutation API'))
    retention.main([])
    assert '"apply": false' in capsys.readouterr().out


def test_partial_old_cleanup_is_resumable_but_newer_incomplete_release_is_preserved():
    old = release('0.1.0')
    old['assets'] = old['assets'][:2]
    newer = release('0.5.0')
    newer['assets'] = newer['assets'][:2]
    plan = retention.plan_release_assets([old, newer, release('0.2.0'), release('0.3.0'), release('0.4.0')])
    assert len(plan) == 2
    assert {asset['tag'] for asset in plan} == {'v0.1.0'}
    assert retention.plan_release_assets([old, release('0.2.0')]) == []


def test_api_rate_limit_honors_retry_after_and_serial_delete_delay(monkeypatch):
    from types import SimpleNamespace

    responses = iter([
        SimpleNamespace(returncode=1, stdout='HTTP/2 429\nretry-after: 120\n\n{}',
                        stderr='gh: rate limited (HTTP 429)'),
        SimpleNamespace(returncode=0, stdout='HTTP/2 204\n\n', stderr=''),
    ])
    waits = []
    monkeypatch.setattr(retention.subprocess, 'run', lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(retention.time, 'sleep', waits.append)
    assert retention.api('repos/owner/repo/releases/assets/1', method='DELETE') is None
    assert waits == [1, 120, 1]


def test_api_failure_stops_instead_of_continuing_deletions(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(retention.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout='', stderr='server error (HTTP 500)'))
    monkeypatch.setattr(retention.time, 'sleep', lambda delay: None)
    with pytest.raises(RuntimeError, match='HTTP 500'):
        retention.api('repos/owner/repo/releases/assets/1', method='DELETE')


def test_workflows_expire_build_outputs_and_run_the_bounded_cleanup():
    import yaml

    root = Path(__file__).resolve().parents[1]
    for filename in ('build-beta-installers.yml', 'publish-pypi.yml'):
        workflow = yaml.safe_load((root / '.github/workflows' / filename).read_text())
        uploads = [step for job in workflow['jobs'].values() for step in job['steps']
                   if str(step.get('uses', '')).startswith('actions/upload-artifact@')]
        assert uploads
        assert all(step['with']['retention-days'] == 7 for step in uploads)
    workflow = yaml.load((root / '.github/workflows/retention.yml').read_text(),
                         Loader=yaml.BaseLoader)
    assert workflow['on']['workflow_run']['workflows'] == ['Build beta installers']
    assert workflow['on']['schedule']
    assert workflow['permissions'] == {'contents': 'write', 'actions': 'write'}
    assert workflow['concurrency']['cancel-in-progress'] == 'false'
    assert any('--apply' in step.get('run', '') for step in workflow['jobs']['cleanup']['steps'])
