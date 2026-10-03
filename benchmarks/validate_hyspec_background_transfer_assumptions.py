"""Controlled detector-background transfer truths; no project or Mantid access.

This checks statistical model assumptions, not production implementation.
Run with the documented nfit interpreter and specify a JSON receipt destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from nfit.dgs_reduction_policy import ENERGY_TO_K


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    seed, repetitions = 73029, 200_000
    rng = np.random.default_rng(seed)
    delta_degrees = 0.026092529296875
    sample_theta = np.array([20., 30.])
    background_theta = sample_theta-delta_degrees
    kf = np.sqrt(15.*ENERGY_TO_K)
    sample_coordinate = kf*np.sin(np.deg2rad(sample_theta))
    background_coordinate = kf*np.sin(np.deg2rad(background_theta))
    lower = (sample_coordinate[0]+background_coordinate[0])/2
    upper = sample_coordinate[1]+.01
    sample_acceptance = (sample_coordinate >= lower)&(sample_coordinate < upper)
    background_acceptance = (background_coordinate >= lower)&(background_coordinate < upper)
    assert sample_acceptance.tolist() == [True, True]
    assert background_acceptance.tolist() == [False, True]

    # Two detector/energy cells with known exposure. Counts are independent
    # Poisson observations; the angular coordinate only changes bin membership.
    sample_charge, background_charge = 400., 100.
    detector_rates = np.array([10., 50.])
    sample_signal = 5.
    sample_exposure = sample_charge*sample_acceptance.sum()
    sample_background = detector_rates[sample_acceptance].mean()
    sample_counts = rng.poisson(sample_exposure*(sample_signal+sample_background), repetitions)
    source_counts = rng.poisson(background_charge*detector_rates, (repetitions, 2))
    source_numerator = source_counts[:, background_acceptance].sum(axis=1)
    remapped_numerator = source_counts[:, sample_acceptance].sum(axis=1)
    choices = {
        'consistent_original_lab_geometry': (
            source_numerator, background_charge*background_acceptance.sum()),
        'mixed_source_numerator_sample_denominator': (
            source_numerator, background_charge*sample_acceptance.sum()),
        'detector_identity_and_energy_remap': (
            remapped_numerator, background_charge*sample_acceptance.sum()),
    }
    result = {}
    for name, (counts, exposure) in choices.items():
        estimates = sample_counts/sample_exposure-counts/exposure
        observed_variance = sample_counts/sample_exposure**2+counts/exposure**2
        sigma = np.sqrt(observed_variance)
        expected_estimate = (
            sample_signal+sample_background
            - float(np.mean(counts))/exposure)
        result[name] = {
            'mean_estimate':float(estimates.mean()),
            'bias_from_known_signal':float(estimates.mean()-sample_signal),
            'monte_carlo_standard_error_of_mean':float(estimates.std(ddof=1)/np.sqrt(repetitions)),
            'empirical_standard_deviation':float(estimates.std(ddof=1)),
            'rms_reported_sigma':float(np.sqrt(observed_variance.mean())),
            'gaussian_95_percent_coverage':float(np.mean(abs(estimates-sample_signal)<=1.96*sigma)),
            'expected_estimate_using_empirical_source_mean':expected_estimate,
        }
    assert abs(result['detector_identity_and_energy_remap']['bias_from_known_signal']) < .005
    assert .945 < result['detector_identity_and_energy_remap']['gaussian_95_percent_coverage'] < .955
    assert abs(result['consistent_original_lab_geometry']['bias_from_known_signal']+20.) < .01
    assert abs(result['mixed_source_numerator_sample_denominator']['bias_from_known_signal']-5.) < .01

    # A different physical hypothesis changes the correct prediction. If the
    # rate is fixed as a laboratory angular field with logarithmic derivative
    # 1/degree, moving the bank changes each pixel's expected background rate.
    lab_field_slope_per_degree = 1.
    field_sample_rates = detector_rates*np.exp(lab_field_slope_per_degree*delta_degrees)
    field_target_background = field_sample_rates[sample_acceptance].mean()
    pixel_transfer_bias = field_target_background-sample_background
    receipt = {
        'seed':seed,'repetitions':repetitions,
        'delta_degrees':delta_degrees,'Ei_meV':15.,'energy_transfer_meV':0.,
        'coordinate':'positive transverse momentum = kf*sin(theta), inverse angstrom',
        'sample_theta_degrees':sample_theta.tolist(),
        'background_theta_degrees':background_theta.tolist(),
        'sample_coordinate':sample_coordinate.tolist(),
        'background_coordinate':background_coordinate.tolist(),
        'bin_bounds':[float(lower),float(upper)],
        'detector_fixed_truth':{
            'sample_signal':sample_signal,'detector_rates':detector_rates.tolist(),
            'sample_charge':sample_charge,'background_charge':background_charge,
            'note':'Independent Poisson counts in detector/energy cells; conditional known exposures; no within-cell energy spread.',
            'comparisons':result},
        'laboratory_field_truth':{
            'logarithmic_rate_derivative_per_degree':lab_field_slope_per_degree,
            'sample_detector_rates':field_sample_rates.tolist(),
            'sample_background_target':float(field_target_background),
            'unadjusted_pixel_transfer_subtraction_bias':float(pixel_transfer_bias),
            'note':'Pixel remapping alone assumes detector-rate stationarity. A laboratory field varying with angle requires a field model/additional observations; consistent original-Qlab replay alone also cannot determine an unmeasured target direction.'},
        'shared_calibration_example':{
            'sample_normalized_rate':35.,'background_normalized_rate':30.,
            'signal':5.,'relative_calibration_sigma':.02,
            'correct_signal_calibration_variance':float((.02*5)**2),
            'incorrect_independent_operand_calibration_variance':float(.02**2*(35**2+30**2)),
            'note':'One multiplicative calibration shared by sample/background partially cancels in their difference; separate independent perturbations destroy that covariance.'},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
