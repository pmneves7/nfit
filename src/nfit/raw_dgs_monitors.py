"""Native monitor peak analysis with Mantid GetEi v2 conventions.

The histogram grid, fractional histogram rebinning, peak background and
half-height widths follow the SNS monitor route; no Mantid runtime is needed.
"""

from __future__ import annotations

import math

import numpy as np

# Mantid PhysicalConstants: microseconds per metre times sqrt(meV).
TOF_US_PER_M_SQRT_MEV = math.sqrt(5e11 * 1.674927211e-27 / 1.602176487e-22)


def _mantid_getei_v2_peak(values, distance, energy_guess, *, grid_start=None):
    """Reproduce Mantid GetEi v2's monitor peak estimate without Mantid."""

    expected = TOF_US_PER_M_SQRT_MEV * distance / math.sqrt(energy_guess)
    lower, upper = 0.9 * expected, 1.1 * expected
    # SNS first rebins all monitor spectra on a common 1-us grid. Cropping
    # keeps complete bins on that grid, rather than restarting at t_min.
    anchor = float(np.min(values)) if grid_start is None else float(grid_start)
    first = anchor + math.ceil(lower - anchor)
    last = anchor + math.floor(upper - anchor)
    if last <= first:
        return None
    initial_edges = np.arange(first, last + 0.5, 1.0)
    counts, edges = np.histogram(values, bins=initial_edges)
    if not np.any(counts):
        return None
    centres = 0.5 * (edges[:-1] + edges[1:])
    region = _mantid_getei_peak_region(centres, counts.astype(float), np.sqrt(counts))
    if region is None:
        return None
    _, _, width = region
    if width <= 0.0:
        return None
    # GetEi serializes the Rebin parameters through a C++ stream with six
    # significant digits, then fractionally rebins the matrix histogram.
    lower, width, upper = (float(format(v, ".6g")) for v in (lower, width / 12.0, upper))
    rebinned_edges = _monitor_rebin_edges(lower, width, upper)
    rebinned, rebinned_errors = _rebin_monitor_histogram(counts, edges, rebinned_edges)
    rebinned_centres = 0.5 * (rebinned_edges[:-1] + rebinned_edges[1:])
    widths = np.diff(rebinned_edges)
    for prominence in (4.0, 2.0):
        region = _mantid_getei_peak_region(
            rebinned_centres, rebinned / widths, rebinned_errors / widths, prominence,
        )
        if region is not None:
            x, y, _ = region
            area = np.trapezoid(y, x)
            if np.isfinite(area) and area > 0.0:
                return float(np.trapezoid(x * y, x) / area)
    return None


def _monitor_rebin_edges(lower, width, upper):
    # Repeated addition matches Mantid's boundary rounding. Tiny changes in
    # empty tail bins can otherwise change its derivative stopping criterion.
    edges = [lower]
    while edges[-1] + width <= upper:
        edges.append(edges[-1] + width)
    edges = np.asarray(edges)
    # Mantid replaces a final bin shorter than 25% of the requested width.
    if upper - edges[-1] < 0.25 * width:
        edges[-1] = upper
    else:
        edges = np.append(edges, upper)
    return edges


def _rebin_monitor_histogram(counts, edges, target):
    """Rebin count integrals and Poisson variances by overlap fraction."""
    rebinned = np.zeros(len(target) - 1)
    i = 0
    for j, (lo, hi) in enumerate(zip(target[:-1], target[1:], strict=True)):
        while i < len(counts) and edges[i + 1] <= lo:
            i += 1
        k = i
        while k < len(counts) and edges[k] < hi:
            overlap = min(hi, edges[k + 1]) - max(lo, edges[k])
            if overlap > 0:
                rebinned[j] += counts[k] * overlap / (edges[k + 1] - edges[k])
            k += 1
    # Mantid propagates Poisson variance by the overlap fraction, not its square.
    return rebinned, np.sqrt(np.maximum(rebinned, 0.0))


def _mantid_getei_peak_region(x, y, errors, prominence=4.0):
    """Port of Mantid GetEi2::calculatePeakWidthAtHalfHeight's peak region."""

    if x.size < 3:
        return None
    peak = int(np.argmax(y))
    background_floor = float(np.min(y))
    peak_height = float(y[peak] - background_floor)
    if peak_height <= 0.0:
        return None
    peak_error = float(errors[peak])
    left = peak - 1
    while left >= 0:
        ratio = (y[left] - background_floor) / peak_height
        ratio_error = math.sqrt(errors[left] ** 2 + (ratio * peak_error) ** 2) / peak_height
        if ratio < 1.0 / prominence - 2.0 * ratio_error:
            break
        left -= 1
    right = peak + 1
    while right < x.size:
        ratio = (y[right] - background_floor) / peak_height
        ratio_error = math.sqrt(errors[right] ** 2 + (ratio * peak_error) ** 2) / peak_height
        if ratio < 1.0 / prominence - 2.0 * ratio_error:
            break
        right += 1
    if left < 0 or right >= x.size:
        return None

    # Match GetEi2's derivative extension of the initially prominent region.
    derivative, uncertainty = -1000.0, 0.0
    while right < x.size - 1 and derivative < -uncertainty:
        forward, backward = x[right + 1] - x[right], x[right] - x[right - 1]
        derivative = 0.5 * (
            (y[right + 1] - y[right]) / forward + (y[right] - y[right - 1]) / backward
        )
        uncertainty = _peak_derivative_uncertainty(
            errors[right - 1], errors[right], errors[right + 1], backward, forward,
        )
        right += 1
    right -= 1
    if derivative < -uncertainty:
        right = x.size - 1

    derivative, uncertainty = 1000.0, 0.0
    while left > 0 and derivative > uncertainty:
        forward, backward = x[left + 1] - x[left], x[left] - x[left - 1]
        derivative = 0.5 * ((y[left + 1] - y[left]) / forward + (y[left] - y[left - 1]) / backward)
        uncertainty = _peak_derivative_uncertainty(
            errors[left - 1], errors[left], errors[left + 1], backward, forward,
        )
        left -= 1
    left += 1
    if derivative > uncertainty:
        left = 0

    peak_width = x[right] - x[left]
    if peak_width <= 0.0:
        return None
    background_start = max(x[0], x[left] - 0.5 * peak_width)
    background_stop = min(x[-1], x[right] + 0.5 * peak_width)
    background_parts = []
    if left > 0:
        background_parts.append((_mantid_point_integral(x, y, background_start, x[left]),
                                 x[left] - background_start))
    if right < x.size - 1:
        background_parts.append((_mantid_point_integral(x, y, x[right], background_stop),
                                 background_stop - x[right]))
    background = (
        sum(area for area, _ in background_parts) / sum(span for _, span in background_parts)
        if background_parts
        else 0.0
    )
    px, py = x[left : right + 1], y[left : right + 1] - background
    half = 0.5 * py[peak - left]
    centre = peak - left
    rising = np.flatnonzero(py[:centre + 1] < half)
    falling = np.flatnonzero(py[centre:] < half) + centre
    lo = 0 if not len(rising) else int(rising[-1])
    hi = len(py) - 1 if not len(falling) else int(falling[0])
    # GetEi brackets crossings from both directions to accommodate local maxima.
    if len(rising):
        lo2 = int(np.flatnonzero(py[:centre + 1] > half)[0]) - 1
        lo1 = lo + 1
        while py[lo1] == py[lo2] and lo1 < centre:
            lo1 += 1
        if py[lo1] == py[lo2]:
            return None
        xmin = px[lo2] + (px[lo1] - px[lo2]) * (half - py[lo2]) / (py[lo1] - py[lo2])
    else:
        xmin = px[0]
    if len(falling):
        hi1 = hi - 1
        hi2 = int(np.flatnonzero(py[centre:] > half)[-1]) + centre + 1
        while py[hi1] == py[hi2] and hi1 > centre:
            hi1 -= 1
        if py[hi1] == py[hi2]:
            return None
        xmax = px[hi2] + (px[hi1] - px[hi2]) * (half - py[hi2]) / (py[hi1] - py[hi2])
    else:
        xmax = px[-1]
    return px, py, xmax - xmin


def _peak_derivative_uncertainty(previous_error, centre_error, next_error, backward, forward):
    """Propagate the centred derivative's shared central measurement once.

    Expanding the central coefficient into separate positive terms and a
    subtraction can produce negative roundoff in sparse monitor tails. Keeping
    its squared coefficient intact preserves the nonnegative variance, including
    unequal spacing and a central error that cancels on a uniform grid.
    """

    return 0.5 * math.sqrt(
        (next_error / forward) ** 2
        + (previous_error / backward) ** 2
        + (centre_error * (1.0 / forward - 1.0 / backward)) ** 2
    )


def _mantid_point_integral(x, y, lower, upper):
    """GetEi's libisis point integration, including its endpoint convention."""
    lo = int(np.searchsorted(x, lower, side="left"))
    hi = int(np.searchsorted(x, upper, side="right")) - 1
    if hi < lo:
        a, b = max(lo - 1, 0), min(hi + 1, len(x) - 1)
        return 0.5 * (upper - lower) / (x[b] - x[a]) * (
            y[b] * (upper + lower - 2 * x[a]) + y[a] * (2 * x[b] - upper - lower))
    xlo, ylo = x[lo], 0.0
    if lo > 0:
        xlo = (lower * (lower - x[lo - 1]) + x[lo - 1] * (x[lo] - lower)) / (x[lo] - x[lo - 1])
        ylo = y[lo - 1] * (x[lo] - lower) / (x[lo] + lower - 2 * x[lo - 1])
    xhi, yhi = x[hi], 0.0
    if hi < len(x) - 1:
        xhi = (upper * (x[hi + 1] - upper) + x[hi + 1] * (upper - x[hi])) / (x[hi + 1] - x[hi])
        yhi = y[hi + 1] * (upper - x[hi]) / (2 * x[hi + 1] - x[hi] - upper)
    return float(np.trapezoid(y[lo:hi + 1], x[lo:hi + 1])
                 + 0.5 * (x[lo] - xlo) * (y[lo] + ylo)
                 + 0.5 * (xhi - x[hi]) * (y[hi] + yhi))
