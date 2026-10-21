"""Shared MACS detector geometry and neutron conversion constants."""

import numpy as np

MACS_E_KSQ_MEV_ANGSTROM2 = 2.0721246
MACS_E_LAMBDA_SQ_MEV_ANGSTROM2 = 81.8042
MACS_DETECTOR_OFFSETS_DEG = np.arange(20, dtype=float) * 8.0
MACS_KIDNEY_ZERO_DEG = -76.0
