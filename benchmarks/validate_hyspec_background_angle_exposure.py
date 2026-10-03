"""Read-only angle-exposure audit of merged HYSPEC dummy-sample MDE files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    for path in sorted(args.directory.glob('*bkg.nxs')):
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        with h5py.File(path, 'r') as handle:
            logs = handle['MDEventWorkspace/experiment0/logs']
            pulse, omega = logs['proton_charge'], logs['omega']
            assert pulse['time'].attrs['start'] == omega['time'].attrs['start']
            assert pulse['value'].attrs['units'] == 'picoCoulombs'
            times, charge = pulse['time'][()], pulse['value'][()]/3.6e9
            angle_times, angles = omega['time'][()], omega['value'][()]
            assert np.all(np.diff(angle_times) >= 0) and np.all(charge >= 0)
            held = angles[np.maximum(np.searchsorted(angle_times, times, side='right')-1, 0)]
            nominal = np.rint(held/18)*18
            unique = np.unique(nominal)
            by_angle = np.array([charge[nominal == angle].sum() for angle in unique])
            fractions = by_angle/by_angle.sum()
            equal = 1/len(unique)
            good_charge = float(logs['gd_prtn_chrg/value'][0])
            source_digest = hashlib.sha256()
            for values in (times, charge, angle_times, angles):
                source_digest.update(values.tobytes())
            sources[path.name] = {
                'nominal_angles_degrees':unique.tolist(),
                'charge_by_angle_microamp_hours':by_angle.tolist(),
                'charge_fractions':fractions.tolist(),
                'equal_angle_fraction':equal,
                'max_relative_deviation_from_equal':float(np.max(abs(fractions/equal-1))),
                'total_variation_distance_to_equal':float(.5*np.sum(abs(fractions-equal))),
                'pulse_charge_total_microamp_hours':float(charge.sum()),
                'stored_good_charge_microamp_hours':good_charge,
                'relative_total_difference':float(charge.sum()/good_charge-1),
                'input_size_and_mtime_ns':list(before),
                'source_log_arrays_sha256':source_digest.hexdigest(),
            }
        assert before == (path.stat().st_size, path.stat().st_mtime_ns)
    assert len(sources) == 4
    receipt = {
        'method':'Pulse-charge log totals assigned by last-held omega, rounded to nominal 18-degree steps; pC converted to microamp hours.',
        'qualification':'Proxy for angle exposure, not exact pause-filtered per-run good-charge reconstruction. Event membership by original dummy angle is absent after QLab merging.',
        'acquisition_context':'User: dummy copy of mount, glue and aluminum without crystals, no can; coarse 180-degree sweep intended to average small orientation-dependent effects.',
        'sources':sources,
        'original_files_unchanged':True,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print(json.dumps({name:{key:value for key,value in item.items() if key in {
        'max_relative_deviation_from_equal', 'total_variation_distance_to_equal',
        'relative_total_difference'}} for name,item in sources.items()}, indent=2))


if __name__ == '__main__':
    main()
