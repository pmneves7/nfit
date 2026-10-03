"""Read-only local HYSPEC acceptance against a saved Mantid histogram.

Run with the documented nfit conda interpreter. This manual diagnostic imports
no Mantid and saves no project. Source MDEs, historical refined UB, actual edges,
unsmoothed background subtraction, and symmetry are taken from stored history.
The default is the identity-symmetry 50 K reference; --symmetry=m-3m is slower.
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

from nfit import (
    mdevent_dataset_group,
    project_measured_background_mdevent,
    replay_measurement_histogram,
)
from nfit.histogram_statistics import selected_event_statistics
from nfit.symmetry import SymmetrySpec, resolve_symmetry

ROOT = Path('/Users/pmneves/Library/CloudStorage/OneDrive-JohnsHopkins/_research/LiV2O4/2026_05_HYSPEC')


def difference(actual, reference, selected=None):
    if selected is not None:
        actual, reference = actual[selected], reference[selected]
    valid = np.isfinite(actual) & np.isfinite(reference)
    a, b = actual[valid], reference[valid]
    nz = b != 0
    return {'cells': int(a.size), 'exact': bool(np.array_equal(actual, reference, equal_nan=True)),
            'finite_support_equal': bool(np.array_equal(np.isfinite(actual), np.isfinite(reference))),
            'max_absolute': float(np.max(abs(a-b), initial=0)),
            'max_relative': float(np.max(abs((a[nz]-b[nz])/b[nz]), initial=0)),
            'rms_absolute': float(np.sqrt(np.mean((a-b)**2))) if len(a) else 0.}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--symmetry', choices=('1', 'm-3m'), default='1')
    args = parser.parse_args()
    reference_path = ROOT/f'histograms/3D_HHL_3meV_50K_metallix_{args.symmetry}.nxs'
    sources = [ROOT/'data'/f'Ei15meV_50K_240Hz_s2_{bank}{suffix}.nxs' for bank in (34,70) for suffix in ('','_bkg')]
    paths = [reference_path, *sources, ROOT/'HYSPEC_all.nfit']
    before = {str(path): [path.stat().st_size,path.stat().st_mtime_ns] for path in paths}
    with h5py.File(reference_path,'r') as handle:
        workspace = handle['MDHistoWorkspace']
        stored_edges = [workspace['data'][f'D{i}'][()] for i in range(4)]
        # Saved coord_t interior edges jitter by float32 ulps; represent the
        # original uniform recipe, not an artificial nonuniform replay grid.
        edges = [np.linspace(float(e[0]),float(e[-1]),len(e)) for e in stored_edges]
        reference_signal = workspace['data/signal'][()].transpose(3,2,1,0)
        reference_variance = workspace['data/errors_squared'][()].transpose(3,2,1,0)
        ub = workspace['experiment0/sample/oriented_lattice/orientation_matrix'][()]
        history = []
        for process in workspace['process'].values():
            text = process['data'][0].decode()
            if text.startswith('Algorithm: MDNorm'):
                history.append(text)
        assert len(history) == 2
        for text in history:
            assert f'Name: SymmetryOperations, Value: {args.symmetry},' in text
            assert 'Name: SolidAngleWorkspace, Value: ,' in text
            assert 'Name: FluxWorkspace, Value: ,' in text
        mantid_environment = workspace['process/MantidEnvironment/data'][0].decode()
    lattice = dict(a=8.227,b=8.227,c=8.227,alpha=90.,beta=90.,gamma=90.)
    operations = [operation.matrix_hkl for operation in resolve_symmetry(SymmetrySpec('point_group',args.symmetry), lattice_parameters=lattice)]
    options = dict(lower=[e[0] for e in edges],upper=[e[-1] for e in edges],num_bins=[len(e)-1 for e in edges],
                   bin_edges=edges,vectors=[[1,1,0,0],[0,0,1,0],[1,-1,0,0],[0,0,0,1]],
                   symmetry_operations=operations,max_batch_bytes=64*1024**2)
    sums = None
    timings, banks = {}, []
    for bank in (34,70):
        sample_path = ROOT/f'data/Ei15meV_50K_240Hz_s2_{bank}.nxs'
        background_path = ROOT/f'data/Ei15meV_50K_240Hz_s2_{bank}_bkg.nxs'
        sample = mdevent_dataset_group(sample_path)
        background = mdevent_dataset_group(background_path)
        sample.metadata['mdevent']['ub_matrix'] = ub.tolist()
        started = time.perf_counter()
        data = replay_measurement_histogram(sample, **options)
        sample_seconds = time.perf_counter()-started
        started = time.perf_counter()
        projected = project_measured_background_mdevent(sample, background, data, max_batch_bytes=64*1024**2)
        background_seconds = time.perf_counter()-started
        # The current replay service exposes normalized background and N,
        # not a certified additive count payload. Reconstruct only a manual
        # diagnostic under its recorded correlated-copy policy; zero-event
        # display confidence intervals are excluded from observed variance.
        background_n = np.asarray(projected.auxiliary_channels["normalization_denominator"].values)
        background_c = np.nan_to_num(projected.signal * background_n)
        background_v = np.where(projected.num_events > 0,
                                np.nan_to_num((projected.errors * background_n)**2), 0.)
        stats = [np.asarray(a) for a in (*selected_event_statistics(data), background_c, background_v, background_n)]
        if sums is None:
            sums = [a.copy() for a in stats]
        else:
            for total, values in zip(sums,stats,strict=True):
                total += values
        banks.append({'s2':bank,'sample_runs':len(sample.datasets),'background_runs':len(background.datasets),
                      'sample_events':sample.metadata['mdevent']['event_count'],'background_events':background.metadata['mdevent']['event_count'],
                      'sample_path':str(sample_path),'background_path':str(background_path),
                      'background_uncertainty_policy':projected.metadata.get('background_projection')})
        timings[str(bank)] = {'sample_histogram_seconds':sample_seconds,'background_replay_seconds':background_seconds}
        print('SAVED_HYSPEC_BANK_COMPLETE',bank,timings[str(bank)],flush=True)
        del data,projected,stats,sample,background
    c,v,n,bc,bv,bn = sums
    with np.errstate(divide='ignore',invalid='ignore'):
        actual_signal = c/n-bc/bn
        actual_variance = v/n**2+bv/bn**2
    covered = (n>0)&(bn>0)&np.isfinite(reference_signal)
    low = covered & (n<=np.quantile(n[covered],.1))
    fringe = covered & binary_dilation(~covered,iterations=2)
    comparisons = {label:{'signal':difference(actual_signal,reference_signal,region),
                         'variance':difference(actual_variance,reference_variance,region),
                         'uncertainty':difference(np.sqrt(actual_variance),np.sqrt(reference_variance),region)}
                   for label,region in (('all',None),('lowest_exposure_decile',low),('coverage_fringe',fringe))}
    after = {str(path): [path.stat().st_size,path.stat().st_mtime_ns] for path in paths}
    assert before == after
    result = {'reference':str(reference_path),'mantid_environment':mantid_environment,
              'UB':ub.tolist(),'actual_edges':[[float(e[0]),float(e[-1]),len(e)-1] for e in edges],
              'shape':list(actual_signal.shape),'symmetry':args.symmetry,'banks':banks,'timings':timings,
              'comparisons':comparisons,'low_exposure_cells':int(low.sum()),'coverage_fringe_cells':int(fringe.sum()),
              'before_inputs':before,'after_inputs':after,'original_inputs_unchanged':True,
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'note':'Native measured background currently pools copies of each source event before variance; compare this explicit covariance policy with historical Mantid variance independently.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(comparisons),flush=True)


if __name__ == '__main__':
    main()
