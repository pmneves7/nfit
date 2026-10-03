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
import re
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_dilation

from nfit import (
    mdevent,
    mdevent_dataset_group,
    project_measured_background_mdevent,
    replay_measurement_histogram,
)
from nfit.histogram_statistics import selected_event_statistics
from nfit.symmetry import SymmetrySpec, resolve_symmetry

ROOT = Path('/Users/pmneves/Library/CloudStorage/OneDrive-JohnsHopkins/_research/LiV2O4/2026_05_HYSPEC')


def historical_sample_membership(histories):
    """Resolve actual sample raw-run membership from GenerateDGSMDE history."""
    membership = {}
    for text in histories:
        if not text.startswith('Algorithm: GenerateDGSMDE'):
            continue
        if 'Name: Type, Value: Data,' not in text:
            continue
        output = re.search(r'Name: OutputWorkspace, Value: (.*?), Default', text).group(1)
        filenames = re.search(r'Name: Filenames, Value: (.*?), Default', text).group(1)
        bank = 70 if output.endswith('_s2_70') else 34
        assert bank not in membership, 'ambiguous historical sample bank'
        membership[bank] = set(re.findall(r'HYS_(\d+)\.nxs\.h5', filenames))
    assert set(membership) == {34, 70}
    return membership


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
        histories = []
        for process in workspace['process'].values():
            text = process['data'][0].decode()
            histories.append(text)
            if text.startswith('Algorithm: MDNorm'):
                history.append(text)
        assert len(history) == 2
        for text in history:
            assert f'Name: SymmetryOperations, Value: {args.symmetry},' in text
            assert 'Name: SolidAngleWorkspace, Value: ,' in text
            assert 'Name: FluxWorkspace, Value: ,' in text
        mantid_environment = workspace['process/MantidEnvironment/data'][0].decode()
    membership = historical_sample_membership(histories)
    lattice = dict(a=8.227,b=8.227,c=8.227,alpha=90.,beta=90.,gamma=90.)
    operations = [operation.matrix_hkl for operation in resolve_symmetry(SymmetrySpec('point_group',args.symmetry), lattice_parameters=lattice)]
    options = dict(lower=[e[0] for e in edges],upper=[e[-1] for e in edges],num_bins=[len(e)-1 for e in edges],
                   bin_edges=edges,vectors=[[1,1,0,0],[0,0,1,0],[1,-1,0,0],[0,0,0,1]],
                   symmetry_operations=operations,max_batch_bytes=64*1024**2)
    sums = None
    historical_background_sums = None
    sample_weighted_background_sums = None
    historical_sample_geometry_normalization = None
    timings, banks = {}, []
    for bank in (34,70):
        sample_path = ROOT/f'data/Ei15meV_50K_240Hz_s2_{bank}.nxs'
        background_path = ROOT/f'data/Ei15meV_50K_240Hz_s2_{bank}_bkg.nxs'
        sample = mdevent_dataset_group(sample_path)
        background = mdevent_dataset_group(background_path)
        original_runs = list(sample.datasets)
        available = {str(run.metadata['run_number']) for run in original_runs}
        assert membership[bank] <= available, 'historical runs absent from source MDE'
        for run in original_runs:
            run.enabled = str(run.metadata['run_number']) in membership[bank]
        selected_runs = [run for run in original_runs if run.enabled]
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
        # A scientifically useful alternative pools each bank's background at
        # its SAMPLE exposure. The historical MDNorm recipe instead pools at
        # background charge times number of selected sample angles.
        sample_n = stats[2]
        projected_signal = np.nan_to_num(projected.signal)
        projected_variance = np.where(projected.num_events > 0,
                                      np.nan_to_num(projected.errors**2), 0.)
        sample_weighted_stats = [projected_signal * sample_n,
                                 projected_variance * sample_n**2]
        if sample_weighted_background_sums is None:
            sample_weighted_background_sums = [a.copy() for a in sample_weighted_stats]
        else:
            for total, values in zip(sample_weighted_background_sums, sample_weighted_stats, strict=True):
                total += values
        # Equal angle weights reproduce the historical signal convention.
        # This preserves native correlated-copy variance; it does NOT emulate
        # Mantid's independent-copy variance and is labeled explicitly below.
        saved_weights = [run.fit_weight for run in selected_runs]
        for run in selected_runs:
            run.fit_weight = 1.0 / float(run.metadata['proton_charge'])
        started = time.perf_counter()
        uniform = project_measured_background_mdevent(sample, background, data,
                                                      max_batch_bytes=64*1024**2)
        uniform_background_seconds = time.perf_counter()-started
        # MDNorm deliberately uses SAMPLE detector trajectories for its
        # background denominator, while native directional replay traces the
        # actual BACKGROUND detector geometry. Isolate that convention here.
        started = time.perf_counter()
        detectors, payloads = mdevent._trajectory_payloads(
            sample, selected_runs, np.linalg.inv(np.asarray(options['vectors'])),
            operations, reference_energy=float(selected_runs[0].metadata['incident_energy']))
        background_charge = sum(float(run.metadata['proton_charge']) for run in background.datasets)
        sample_geometry_payloads = [
            (inverse, ei, bounds, background_charge, geometry)
            for inverse, ei, bounds, _, geometry in payloads]
        strict_background_n = mdevent._trajectory_normalization_from_payloads(
            detectors, sample_geometry_payloads, edges, data.shape,
            mantid_precision=True)
        strict_normalization_seconds = time.perf_counter()-started
        if historical_sample_geometry_normalization is None:
            historical_sample_geometry_normalization = strict_background_n.copy()
        else:
            historical_sample_geometry_normalization += strict_background_n
        for run, weight in zip(selected_runs, saved_weights, strict=True):
            run.fit_weight = weight
        angle_count = len(selected_runs)
        uniform_n = np.asarray(uniform.auxiliary_channels['normalization_denominator'].values) * angle_count
        uniform_c = np.nan_to_num(uniform.signal * uniform_n)
        uniform_v = np.where(uniform.num_events > 0,
                             np.nan_to_num((uniform.errors * uniform_n)**2), 0.)
        historical_stats = [uniform_c, uniform_v, uniform_n]
        if historical_background_sums is None:
            historical_background_sums = [a.copy() for a in historical_stats]
        else:
            for total, values in zip(historical_background_sums, historical_stats, strict=True):
                total += values
        if sums is None:
            sums = [a.copy() for a in stats]
        else:
            for total, values in zip(sums,stats,strict=True):
                total += values
        banks.append({'s2':bank,'sample_runs':len(sample.datasets),'background_runs':len(background.datasets),
                      'selected_sample_runs':angle_count,
                      'excluded_sample_runs':sorted(available-membership[bank], key=int),
                      'selected_sample_charge':sum(float(run.metadata['proton_charge']) for run in selected_runs),
                      'original_sample_charge':sum(float(run.metadata['proton_charge']) for run in original_runs),
                      'background_charge':sum(float(run.metadata['proton_charge']) for run in background.datasets),
                      'sample_events':sample.metadata['mdevent']['event_count'],'background_events':background.metadata['mdevent']['event_count'],
                      'sample_path':str(sample_path),'background_path':str(background_path),
                      'background_uncertainty_policy':projected.metadata.get('background_projection')})
        timings[str(bank)] = {'sample_histogram_seconds':sample_seconds,'background_replay_seconds':background_seconds,
                             'uniform_background_replay_seconds':uniform_background_seconds,
                             'historical_sample_geometry_normalization_seconds':strict_normalization_seconds}
        print('SAVED_HYSPEC_BANK_COMPLETE',bank,timings[str(bank)],flush=True)
        del data,projected,uniform,stats,sample,background
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
    hbc,hbv,hbn = historical_background_sums
    swbc,swbv = sample_weighted_background_sums
    with np.errstate(divide='ignore',invalid='ignore'):
        historical_signal = c/n-hbc/hbn
        historical_variance_correlated = v/n**2+hbv/hbn**2
        sample_weighted_signal = (c-swbc)/n
        sample_weighted_variance = (v+swbv)/n**2
        strict_geometry_signal = c/n-hbc/historical_sample_geometry_normalization
    alternative_comparisons = {
        'historical_sample_detector_geometry': {
            'signal':difference(strict_geometry_signal,reference_signal),
            'note':'Historical uniform replay numerator with sample detector geometry and background charge per angle denominator; native correlated variance is not compared.'},
        'historical_uniform_angles_background_charge_pool': {
            'variance_policy': 'native correlated copies; not Mantid independent copies',
            'signal': difference(historical_signal,reference_signal),
            'variance': difference(historical_variance_correlated,reference_variance)},
        'native_sample_exposure_bank_pool': {
            'variance_policy': 'native correlated copies within bank, independent source banks',
            'signal': difference(sample_weighted_signal,reference_signal),
            'variance': difference(sample_weighted_variance,reference_variance)}}
    residual = np.where(np.isfinite(actual_signal)&np.isfinite(reference_signal),
                        abs(actual_signal-reference_signal), -np.inf)
    worst = np.argsort(residual.ravel())[-12:][::-1]
    worst_cells = []
    for flat in worst:
        index = np.unravel_index(int(flat), actual_signal.shape)
        worst_cells.append({
            'index': [int(i) for i in index],
            'centers': [float((edge[i]+edge[i+1])/2) for edge,i in zip(edges,index,strict=True)],
            'native_signal':float(actual_signal[index]),
            'historical_signal':float(reference_signal[index]),
            'historical_uniform_angle_signal':float(historical_signal[index]),
            'historical_sample_geometry_signal':float(strict_geometry_signal[index]),
            'historical_sample_geometry_background_N':float(historical_sample_geometry_normalization[index]),
            'historical_uniform_background_C':float(hbc[index]),
            'background_V':float(bv[index]),
            'native_sample_signal':float(c[index]/n[index]),
            'native_background_signal':float(bc[index]/bn[index]),
            'sample_C':float(c[index]),'sample_V':float(v[index]),'sample_N':float(n[index]),
            'background_C':float(bc[index]),'background_N':float(bn[index]),
            'historical_variance':float(reference_variance[index])})
    after = {str(path): [path.stat().st_size,path.stat().st_mtime_ns] for path in paths}
    assert before == after
    result = {'reference':str(reference_path),'mantid_environment':mantid_environment,
              'UB':ub.tolist(),'actual_edges':[[float(e[0]),float(e[-1]),len(e)-1] for e in edges],
              'shape':list(actual_signal.shape),'symmetry':args.symmetry,'banks':banks,'timings':timings,
              'comparisons':comparisons,'low_exposure_cells':int(low.sum()),'coverage_fringe_cells':int(fringe.sum()),
              'alternative_comparisons':alternative_comparisons,
              'worst_cells':worst_cells,
              'before_inputs':before,'after_inputs':after,'original_inputs_unchanged':True,
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'note':'Native measured background currently pools copies of each source event before variance; compare this explicit covariance policy with historical Mantid variance independently.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(comparisons),flush=True)


if __name__ == '__main__':
    main()
