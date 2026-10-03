"""Read-only HYSPEC directional replay and final-bin covariance diagnostic.

This manual diagnostic imports no Mantid. It reads complete real background MDE
files and nine evenly spaced sample angles per bank. The independent event oracle
uses direct lab-frame transformations and explicit pairwise covariance. Only
scalar receipts are saved; all scientific inputs remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_dilation

from nfit import mdevent_dataset_group, project_measured_background_mdevent
from nfit.mdhisto import MDHistoAxis, MDHistoData

ROOT = Path('/Users/pmneves/Library/CloudStorage/OneDrive-JohnsHopkins/_research/LiV2O4/2026_05_HYSPEC')
BASIS = np.array([[1, 1, 0, 0], [0, 0, 1, 0], [1, -1, 0, 0], [0, 0, 0, 1.]])


def target(edges):
    shape = tuple(len(edge)-1 for edge in edges)
    return MDHistoData(
        axes=tuple(MDHistoAxis(name, edge, 'meV' if i == 3 else 'r.l.u.',
                              'energy' if i == 3 else 'momentum')
                   for i, (name, edge) in enumerate(zip(('HH0', '00L', 'H-H0', 'DeltaE'), edges, strict=True))),
        signal=np.zeros(shape), errors=np.zeros(shape), mask=np.zeros(shape, dtype=bool),
        num_events=np.zeros(shape), metadata={'rebin': {'vectors': BASIS.tolist()}},
    )


def compare(actual, expected, selected=None):
    if selected is not None:
        actual, expected = actual[selected], expected[selected]
    delta = np.asarray(actual)-np.asarray(expected)
    scale = float(np.max(np.abs(expected), initial=0))
    maximum = float(np.max(np.abs(delta), initial=0))
    assert maximum <= 2e-11*max(scale, 1.), (maximum, scale)
    return {'cells': int(delta.size), 'max_absolute': maximum,
            'max_relative_to_peak': maximum/scale if scale else 0.}


def parameter_masks(experiment):
    """Independent parser for the real files' serialized boolean masks."""
    raw = experiment['instrument/instrument_parameter_map/data'][()]
    text = ''.join(v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(raw).ravel())
    state = {}
    for record in text.split('|'):
        fields = record.split(';')
        if len(fields) > 3 and fields[1:3] == ['bool', 'masked'] and fields[0].startswith('detID:'):
            state[int(fields[0].split(':')[1])] = fields[3] in ('1', 'true')
    return [key for key, masked in state.items() if masked]


def event_oracle(sample, background, runs, edges, regions):
    shape = tuple(len(edge)-1 for edge in edges)
    size = int(np.prod(shape))
    c, v, count = (np.zeros(size) for _ in range(3))
    energy_bins = shape[-1]
    region_v = {name: np.zeros(energy_bins) for name in regions}
    fractions = np.array([r.metadata['proton_charge']*r.fit_weight for r in runs], dtype=float)
    fractions /= fractions.sum()
    ub = np.asarray(sample.metadata['mdevent']['ub_matrix'])
    inverses, bounds, masked = [], [], []
    with h5py.File(runs[0].metadata['source_file']) as handle:
        for run in runs:
            exp = handle[f"MDEventWorkspace/experiment{run.metadata['mdevent_experiment_index']}"]
            gonio = exp['logs/goniometer/rotation_matrix'][()].reshape(3, 3)
            inverses.append(np.linalg.inv(BASIS)[:3, :3].T @ np.linalg.inv(gonio @ (2*np.pi*ub)))
            bounds.append(exp['logs/processed_histogram_bins/value'][()][[0, -1]])
            masked.append(parameter_masks(exp))
    source = background.datasets[0]
    assert source.fit_weight == source.scale_factor == 1
    with h5py.File(source.metadata['source_file']) as handle:
        exp = handle['MDEventWorkspace/experiment0']
        source_bounds = exp['logs/processed_histogram_bins/value'][()][[0, -1]]
        source_masks = parameter_masks(exp)
        gonio = exp['logs/goniometer/rotation_matrix'][()].reshape(3, 3)
        values = handle['MDEventWorkspace/event_data/event_data']
        assert background.metadata['mdevent']['dimensions'][0]['frame'] in ('QLab', 'QSample')
        for start in range(0, len(values), 32768):
            block = values[start:start+32768].astype(float)
            assert np.all(block[:, 2:4] == 0)
            lab = block[:, 5:8]
            if background.metadata['mdevent']['dimensions'][0]['frame'] == 'QSample':
                lab = lab @ gonio.T
            locations = []
            for inverse, limit, masks in zip(inverses, bounds, masked, strict=True):
                coords = np.column_stack((lab @ inverse.T, block[:, 8]))
                indices = np.empty((len(block), 4), dtype=np.int64)
                accepted = (~np.isin(block[:, 4], [*source_masks, *masks])
                            & (block[:, 8] >= max(source_bounds[0], limit[0]))
                            & (block[:, 8] <= min(source_bounds[1], limit[1])))
                for dim, edge in enumerate(edges):
                    index = np.searchsorted(edge, coords[:, dim], side='right')-1
                    index[coords[:, dim] == edge[-1]] = len(edge)-2
                    accepted &= (index >= 0) & (index < shape[dim])
                    indices[:, dim] = np.clip(index, 0, shape[dim]-1)
                flat = np.ravel_multi_index(indices.T, shape)
                locations.append(np.where(accepted, flat, -1))
            locations = np.array(locations).T
            for angle, fraction in enumerate(fractions):
                bins = locations[:, angle]
                good = bins >= 0
                c += np.bincount(bins[good], weights=block[good, 0]*fraction, minlength=size)
                v += np.bincount(bins[good], weights=block[good, 1]*fraction**2, minlength=size)
                first = good & ~np.any(locations[:, :angle] == bins[:, None], axis=1)
                count += np.bincount(bins[first], minlength=size)
                for earlier in range(angle):
                    same = good & (bins == locations[:, earlier])
                    v += np.bincount(bins[same], weights=block[same, 1]*2*fraction*fractions[earlier], minlength=size)
            energy_index = np.clip(np.searchsorted(edges[-1], block[:, 8], side='right')-1, 0, energy_bins-1)
            for name, region in regions.items():
                valid = locations >= 0
                coefficients = np.sum(fractions[None, :]*valid*region.ravel()[np.maximum(locations, 0)], axis=1)
                region_v[name] += np.bincount(energy_index, weights=block[:, 1]*coefficients**2, minlength=energy_bins)
    return c.reshape(shape), v.reshape(shape), count.reshape(shape), region_v


def bank_check(temperature, bank):
    paths = [ROOT/'data'/f'Ei15meV_{temperature}K_240Hz_s2_{bank}.nxs',
             ROOT/'data'/f'Ei15meV_{"1p8" if temperature == "1p5" else temperature}K_240Hz_s2_{bank}_bkg.nxs']
    sample, background = map(mdevent_dataset_group, paths)
    runs = [sample.datasets[i] for i in np.linspace(0, len(sample.datasets)-1, 9, dtype=int)]
    # A bounded interior slab with resolved transverse bins and real fringes.
    edges = [np.linspace(-2, 2, 81), np.linspace(-1, 1, 21), np.linspace(-.5, .62, 9), np.linspace(-12, 12, 49)]
    fine = project_measured_background_mdevent(sample, background, target(edges), datasets=runs)
    n = np.asarray(fine.auxiliary_channels['normalization_denominator'].values)
    covered = n > 0
    regions = {'all_covered': covered,
               'lowest_exposure_decile': covered & (n <= np.quantile(n[covered], .1)),
               'coverage_fringe': covered & binary_dilation(~covered, iterations=2)}
    c, v, counts, region_v = event_oracle(sample, background, runs, edges, regions)
    occupied = covered & (counts > 0)
    # Zero-count display intervals are distinct from accumulated event variance.
    actual_c = np.nan_to_num(fine.signal*n)
    actual_v = np.where(occupied, np.nan_to_num((fine.errors*n)**2), 0.)
    checks = {name: {'numerator': compare(actual_c, c, region),
                     'variance': compare(actual_v, v, region),
                     'source_event_counts': compare(fine.num_events, counts, region)}
              for name, region in regions.items()}
    coarse_edges = [e[[0, -1]] if i < 3 else e for i, e in enumerate(edges)]
    coarse = project_measured_background_mdevent(sample, background, target(coarse_edges), datasets=runs)
    cn = coarse.auxiliary_channels['normalization_denominator'].values.ravel()
    cv = np.where(coarse.num_events.ravel() > 0, np.nan_to_num((coarse.errors.ravel()*cn)**2), 0.)
    final_checks = {'numerator': compare(np.nan_to_num(coarse.signal.ravel()*cn), c.sum(axis=(0, 1, 2))),
                    'variance': compare(cv, region_v['all_covered']),
                    'exposure_additivity': compare(cn, n.sum(axis=(0, 1, 2)))}
    ratios = {}
    for name, region in regions.items():
        diagonal = np.sum(np.where(region, v, 0.), axis=(0, 1, 2))
        good = diagonal > 0
        ratio = np.sqrt(region_v[name][good]/diagonal[good])
        ratios[name] = {'selected_cells': int(region.sum()),
                        'sigma_ratio_exact_to_diagonal_min': float(np.min(ratio, initial=np.inf)),
                        'sigma_ratio_exact_to_diagonal_max': float(np.max(ratio, initial=0)),
                        'sigma_ratio_exact_to_diagonal_median': float(np.median(ratio))}
    return {'temperature_K': temperature, 'bank': bank, 'background_events': background.metadata['mdevent']['event_count'],
            'sample_runs': [r.metadata['run_number'] for r in runs], 'sample_charge_uAh': [r.metadata['proton_charge'] for r in runs],
            'fine_shape': list(fine.shape), 'comparisons': checks, 'direct_final_grid': final_checks,
            'covariance_effect_on_energy_profile': ratios, 'covered_zero_count_bins': int(np.count_nonzero(covered & (counts == 0))),
            'coverage_support_equal': bool(np.array_equal(~fine.mask, covered)),
            'background_frame': background.metadata['mdevent']['dimensions'][0]['frame']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = [ROOT/'HYSPEC_all.nfit', *(ROOT/'data'/f'Ei15meV_{t}K_240Hz_s2_{b}{s}.nxs'
              for t, b in (('50', 34), ('50', 70), ('1p8', 34), ('1p8', 70)) for s in ('', '_bkg'))]
    paths.remove(ROOT/'data'/'Ei15meV_1p8K_240Hz_s2_70.nxs')
    paths.append(ROOT/'data'/'Ei15meV_1p5K_240Hz_s2_70.nxs')
    before = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
    results = []
    for temperature, bank in (('50', 34), ('50', 70), ('1p8', 34), ('1p5', 70)):
        started = time.perf_counter()
        result = bank_check(temperature, bank)
        result['seconds'] = time.perf_counter()-started
        results.append(result)
        print('HYSPEC_DIRECTIONAL_REPLAY_VALIDATED', temperature, bank, result['seconds'], flush=True)
    after = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
    assert before == after
    assert all(row['coverage_support_equal'] for row in results)
    output = {'banks': results, 'original_inputs_unchanged': True, 'before_inputs': before, 'after_inputs': after,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Complete measured backgrounds at nine selected sample angles; identity symmetry, unity normalization, no user masks. Event oracle is independent of replay accumulation; exposure additivity uses the existing trajectory integrator, independently validated against Mantid in 6A-R.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
