"""Bounded CORELLI stage-6 acceptance; run with the existing ORNL nfit runtime.

Original projects are read only. A deterministic first-4096-events-per-bank
probe is stored in the requested IPTS work directory with original metadata and
calibrations. Keeping full-run charge intentionally means its intensity is not
an estimator of the full run: this probe tests reconstruction/replay parity,
not a throughput benchmark or scientific estimate from an unbiased sample.

The Python and compiled physics implementations are compared on the same probe.
An independent overlapping-copy reference checks diagonal variance. Energy
hypotheses and symmetry copies reuse physical events; no covariance is inferred.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import platform
import socket
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from nfit import bin_corelli_group, corelli_dataset_group, load_project
from nfit import corelli as corelli_module
from nfit.measurement_contracts import MeasurementContract
from nfit.measurement_diagnostics import available_confidence_channels
from nfit.measurement_regions import estimate_measurement_region
from nfit.reduction_recipes import export_reduction_recipe, replay_reduction_recipe

BASE = Path('/SNS/users/paulneves/data/SNS/CORELLI/IPTS-32964/shared/nfit')
DEFAULT_PROJECT = BASE / 'YMn2_150K_CORELLI_403634-403669_10percent.nfit'


def make_probe(source, target, per_bank=4096):
    """Retain deterministic source events and untouched pulse/geometry metadata."""
    count = 0
    with h5py.File(source, 'r') as original, h5py.File(target, 'w') as probe:
        entry = probe.create_group('entry')
        for key, value in original['entry'].attrs.items():
            entry.attrs[key] = value
        for name, obj in original['entry'].items():
            if not (name.startswith('bank') and name.endswith('_events') and 'event_id' in obj):
                original.copy(obj, entry, name=name)
                continue
            bank = entry.create_group(name)
            for key, value in obj.attrs.items():
                bank.attrs[key] = value
            kept = min(per_bank, obj['event_id'].size)
            count += kept
            for key, item in obj.items():
                if key in {'event_id', 'event_time_offset'}:
                    result = bank.create_dataset(key, data=item[:kept])
                    for attr, value in item.attrs.items():
                        result.attrs[attr] = value
                elif key == 'event_index':
                    result = bank.create_dataset(key, data=np.minimum(item[()], kept))
                    for attr, value in item.attrs.items():
                        result.attrs[attr] = value
                else:
                    original.copy(item, bank, name=key)
    return count


def comparison(a, b):
    checks = {}
    for name in ('signal', 'errors', 'num_events'):
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        checks[name] = {'allclose': bool(np.allclose(x, y, rtol=2e-11, atol=1e-15)),
                        'max_abs_error': float(np.max(np.abs(x-y), initial=0.0))}
    checks['mask_equal'] = bool(np.array_equal(a.mask, b.mask))
    checks['passed'] = checks['mask_equal'] and all(checks[name]['allclose'] for name in ('signal', 'errors', 'num_events'))
    assert checks['passed'], f'CORELLI numerical parity failed: {checks}'
    return checks


def independent_primitive_reference(seed=32964, repetitions=200_000):
    """Repeated acquisition references for positively/negatively correlated channels.

    Independent measured counts X have Var(X)=mean(X). Reconstruction W X
    has covariance W diag(mean(X)) W.T. A final-channel sum must combine each
    primitive coefficient first; neither signed channels nor copies are counts.
    """
    mean = np.array([8., 1., 20., 2.])
    measured = np.random.default_rng(seed).poisson(mean, size=(repetitions, len(mean)))
    cases = {}
    for name, coefficients in (
        ('positive_channel_correlation', np.array([[1., -.5, .2, 0.], [1., .5, -.2, 1.]])),
        ('negative_channel_correlation', np.array([[1., -.5, .2, 0.], [-1., -.5, -.2, 1.]])),
    ):
        values = measured @ coefficients.T
        exact_covariance = (coefficients * mean) @ coefficients.T
        exact_final_variance = float(np.sum(mean * np.sum(coefficients, axis=0)**2))
        observed_final_variance = float(np.var(np.sum(values, axis=1), ddof=1))
        relative_error = abs(observed_final_variance / exact_final_variance - 1.)
        cases[name] = {
            'exact_channel_covariance': exact_covariance.tolist(),
            'naive_diagonal_final_variance': float(np.trace(exact_covariance)),
            'correct_final_variance': exact_final_variance,
            'repeated_sampling_final_variance': observed_final_variance,
            'relative_error': relative_error,
            'passed': relative_error < .025,
        }
    return {'seed': seed, 'repetitions': repetitions, 'cases': cases,
            'passed': all(case['passed'] for case in cases.values())}


def main():
    work = Path(os.environ.get('NFIT_CORELLI_ACCEPTANCE_WORK', str(BASE / '.stage6a-work')))
    work.mkdir(parents=True, exist_ok=True)
    output = work / 'measurement-acceptance-corelli.json'
    output.unlink(missing_ok=True)
    before = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in BASE.glob('*.nfit')}
    started = perf_counter()
    project = load_project(DEFAULT_PROJECT)
    lazy_seconds = perf_counter()-started
    original = next(g for root in project.data_groups for g in root.iter_subgroups()
                    if g.metadata.get('raw_dgs', {}).get('format') == 'corelli-correlation-nexus')
    source = Path(original.datasets[0].metadata['source_file'])
    probe = work / 'corelli-stage6a-first4096-per-bank.nxs.h5'
    events = make_probe(source, probe)
    config = original.metadata['raw_dgs']
    group = corelli_dataset_group([probe], solid_angle_path=config.get('normalization_file'),
                                 flux_path=config.get('flux_file'), mask_path=config.get('mask_file'), name='CORELLI stage6A probe')
    group.metadata['raw_dgs'].update({key: copy.deepcopy(value) for key, value in config.items()
                                    if key not in {'source_files', 'event_count'}})
    grid = dict(lower=[-8., -8., -8., -1.5], upper=[8., 8., 8., 1.5], num_bins=[8, 8, 8, 3],
                fractional_axes=[False]*4, max_batch_bytes=16*1024**2)
    recipe = export_reduction_recipe(group)
    replay = replay_reduction_recipe(recipe)
    modules = {}
    for name in ('corelli', '_corelli_numba', 'corelli_constants', 'raw_dgs', 'reduction_recipes'):
        module = importlib.import_module('nfit.'+name)
        modules[name] = {'path': module.__file__, 'sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
    source_commit = os.environ.get('NFIT_ACCEPTANCE_COMMIT')
    if not source_commit:
        raise RuntimeError('Set NFIT_ACCEPTANCE_COMMIT to identify the scientific source build')
    results = {'host': socket.gethostname(), 'python': platform.python_version(), 'numpy': np.__version__,
               'source_commit': source_commit, 'source_modules': modules,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'schema_version': 1, 'instrument': 'CORELLI', 'project': str(DEFAULT_PROJECT),
               'original_run': str(source), 'sample_events': events, 'sample_policy': 'first 4096 events per bank; original full-run charge',
               'project_reopen_seconds': lazy_seconds, 'grid': {k:v for k,v in grid.items() if k != 'max_batch_bytes'},
               'checks': {}, 'timings_seconds': {}, 'limitations': [],
               'timing_context': 'Same process after probe reductions: full-run timing is warmed for JIT, geometry and filesystem cache; only three energy hypotheses.',
               'acceptance_status': 'Numerical reconstruction/replay parity passed; physical final-cut covariance and trajectory-coverage acceptance incomplete.'}
    backend = corelli_module._CORELLI_NUMBA
    if backend is None:
        raise RuntimeError('CORELLI acceptance requires the existing compiled backend')
    compiled_calls = []
    run_hkle = backend.run_hkle

    def counted_run_hkle(*args, **kwargs):
        compiled_calls.append(True)
        return run_hkle(*args, **kwargs)

    backend.run_hkle = counted_run_hkle
    for mode in ('discrete', 'fractional'):
        requested = dict(grid, fractional_axes=[mode == 'fractional']*3+[False])
        compiled_calls.clear()
        t = perf_counter()
        compiled = bin_corelli_group(group, **requested)
        results['checks'][mode+'_compiled_backend_exercised'] = bool(compiled_calls)
        assert compiled_calls, f'{mode} probe did not exercise compiled CORELLI reconstruction'
        results['timings_seconds'][mode+'_compiled_including_first_jit'] = perf_counter()-t
        corelli_module._CORELLI_NUMBA = None
        try:
            t = perf_counter()
            python = bin_corelli_group(group, **requested)
            results['timings_seconds'][mode+'_python'] = perf_counter()-t
        finally:
            corelli_module._CORELLI_NUMBA = backend
        results['checks'][mode+'_compiled_python'] = comparison(python, compiled)
        results['checks'][mode+'_confidence_channels_absent'] = not available_confidence_channels(compiled)
        assert results['checks'][mode+'_confidence_channels_absent'], 'Signed CORELLI channels received unjustified Poisson intervals'
        if mode == 'discrete':
            replayed = bin_corelli_group(replay, **requested)
            results['checks']['portable_recipe_replay'] = comparison(compiled, replayed)
            results['signed_bins'] = int(np.count_nonzero((compiled.signal < 0) & ~compiled.mask))
            results['occupied_bins'] = int(np.count_nonzero(~compiled.mask))
            results['reconstruction_metadata'] = compiled.metadata['corelli_reconstruction']
    symmetry_grid = dict(grid, symmetry_operations=[np.eye(3), -np.eye(3)])
    t = perf_counter()
    sym = bin_corelli_group(group, **symmetry_grid)
    results['timings_seconds']['symmetry_compiled'] = perf_counter()-t
    corelli_module._CORELLI_NUMBA = None
    try:
        sym_python = bin_corelli_group(group, **symmetry_grid)
    finally:
        corelli_module._CORELLI_NUMBA = backend
    results['checks']['symmetry_compiled_python'] = comparison(sym_python, sym)
    # One Q cell makes inversion copies coincide exactly: sum(w+w)^2=4w².
    single = dict(grid, num_bins=[1, 1, 1, 3])
    plain = bin_corelli_group(group, **single)
    duplicated = bin_corelli_group(group, **dict(single, symmetry_operations=[np.eye(3), -np.eye(3)]))
    finite = (~plain.mask) & (plain.errors > 0)
    results['symmetry_overlap_reference'] = {
        'signal_double_max_abs_error': float(np.max(np.abs(duplicated.signal-2*plain.signal))),
        'reported_variance_ratio': (duplicated.errors[finite]**2 / plain.errors[finite]**2).tolist(),
        'physical_shared_event_variance_ratio': 4.0,
        'status': 'limitation: symmetry copies counted as independent in diagonal variance',
    }
    contract = MeasurementContract(kind='linear_reconstruction', estimator='linear_sum',
                                  quantity='CORELLI energy-channel integral', value_units='arbitrary units',
                                  dependence='shared_sources', missing='omit')
    try:
        estimate_measurement_region(plain.signal, plain.errors, mask=plain.mask, contract=contract)
    except ValueError as exc:
        results['checks']['explicit_shared_region_rejected_without_covariance'] = str(exc)
    else:
        raise AssertionError('Shared CORELLI region was accepted without covariance or source replay')
    full_group = corelli_dataset_group([source], solid_angle_path=config.get('normalization_file'),
                                      flux_path=config.get('flux_file'), mask_path=config.get('mask_file'), name='CORELLI full-run acceptance')
    full_group.metadata['raw_dgs'].update({key: copy.deepcopy(value) for key, value in config.items()
                                         if key not in {'source_files', 'event_count'}})
    t = perf_counter()
    full = bin_corelli_group(full_group, **grid)
    results['timings_seconds']['unmodified_full_run_compiled'] = perf_counter()-t
    results['full_run'] = {
        'source_events': full_group.datasets[0].metadata['event_count'],
        'occupied_bins': int(np.count_nonzero(~full.mask)),
        'signed_bins': int(np.count_nonzero((full.signal < 0) & ~full.mask)),
        'signal_sum': float(np.sum(full.signal)),
        'diagonal_variance_sum': float(np.sum(full.errors[~full.mask]**2)),
        'reconstruction_metadata': full.metadata['corelli_reconstruction'],
        'finite_unmasked_values': bool(np.all(np.isfinite(full.signal[~full.mask])) and np.all(np.isfinite(full.errors[~full.mask]))),
        'confidence_channels_absent': not available_confidence_channels(full),
    }
    assert results['full_run']['finite_unmasked_values'], 'Full-run CORELLI histogram contains invalid unmasked values'
    assert results['full_run']['confidence_channels_absent'], 'Full-run signed channels received unjustified Poisson intervals'
    results['independent_primitive_reference'] = independent_primitive_reference()
    assert results['independent_primitive_reference']['passed'], 'Repeated-acquisition variance reference failed'
    results['limitations'] = [
        'Finite-energy CORELLI is not Mantid elastic CorelliCrossCorrelate equivalence; no Mantid parity claim here.',
        'Energy hypotheses from one neutron share covariance; source-level final-cut covariance is not retained by current CORELLI adapter.',
        'Pointwise solid-angle/flux correction and charge/duty normalization are not full 4D MDNorm trajectory normalization.',
        'Probe timings include metadata/calibration reads and possible first JIT; not full-run throughput timings.',
        'No exact Poisson likelihood/confidence interval is justified for signed reconstructed intensity.',
    ]
    backend.run_hkle = run_hkle
    after = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in BASE.glob('*.nfit')}
    results['checks']['original_projects_unchanged'] = before == after
    assert before == after, 'Original CORELLI scientific project changed during validation'
    results['project_size_mtime'] = before
    results['elapsed_seconds'] = perf_counter()-started
    output.write_text(json.dumps(results, indent=2, allow_nan=False))
    print(json.dumps({'output': str(output), 'events':events, 'checks':results['checks'],
                      'symmetry_overlap_reference': results['symmetry_overlap_reference'],
                      'timings': results['timings_seconds']}, default=str), flush=True)


if __name__ == '__main__':
    main()
