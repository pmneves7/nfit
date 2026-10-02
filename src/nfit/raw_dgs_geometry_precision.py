"""Source-ordered Mantid He-3 geometry arithmetic for supported cylinders.

This reproduces the radius obtained by Mantid's distant cardinal ray for an
unrotated cardinal cylinder whose transverse center is the origin. It is not
a general constructive-solid-geometry intersection service. All lengths are
in metres; the returned exponential coefficient is per angstrom.
"""

from __future__ import annotations

import math

import numpy as np

HE3_EFFICIENCY_EXPONENTIAL_CONSTANT = 2175.486863864
_RAY_DISTANCE = 1000.0


def _dot(left, right):
    # V3D::scalar_prod evaluates these scalar terms in this order.
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _vector(value):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        return None
    return tuple(float(item) for item in array)


def mantid_cylinder_radius(radius, axis, centre_bottom, height):
    """Return Mantid's ray-derived radius, or ``None`` for unsupported shapes.

    ``axis`` and ``centre_bottom`` are local shape coordinates, before the
    physical detector rotation. The bottom base must include the origin's
    axial position, and its transverse displacement must be exactly zero.
    Missing height, non-cardinal axes, and invalid geometry are unsupported.
    Callers decide whether unsupported geometry should fail or use a declared
    compatibility fallback.
    """
    axis = _vector(axis)
    bottom = _vector(centre_bottom)
    try:
        radius, height = float(radius), float(height)
    except (TypeError, ValueError):
        return None
    if (axis is None or bottom is None or not math.isfinite(radius)
            or not math.isfinite(height) or radius <= 0.0 or height <= 0.0):
        return None
    length = math.sqrt(_dot(axis, axis))
    if length <= 0.0:
        return None
    axis = tuple(value / length for value in axis)
    axial_dimension = max(range(3), key=lambda index: abs(axis[index]))
    # A nearly cardinal axis is still a different shape. Do not hide a tilt.
    if (abs(axis[axial_dimension]) != 1.0
            or any(axis[index] != 0.0 for index in range(3) if index != axial_dimension)
            or any(bottom[index] != 0.0 for index in range(3) if index != axial_dimension)):
        return None
    axial_bottom = _dot(bottom, axis)
    if not axial_bottom <= 0.0 <= axial_bottom + height:
        return None
    center = tuple(bottom[index] + axis[index] * (0.5 * height) for index in range(3))
    ray_dimension = 1 if axial_dimension == 0 else 0
    origin = tuple(_RAY_DISTANCE if index == ray_dimension else 0.0 for index in range(3))
    direction = tuple(-1.0 if index == ray_dimension else 0.0 for index in range(3))
    relative = tuple(origin[index] - center[index] for index in range(3))
    direction_projection = _dot(axis, direction)
    center_projection = _dot(axis, relative)
    a = 1.0 - direction_projection * direction_projection
    b = 2.0 * (_dot(relative, direction) - center_projection * direction_projection)
    c = _dot(relative, relative) - (radius * radius + center_projection * center_projection)
    discriminant = b * b - 4.0 * a * c
    if discriminant <= 0.0:
        return None
    root = math.sqrt(discriminant)
    q = -0.5 * (b + root) if b >= 0.0 else -0.5 * (b - root)
    if q == 0.0:
        return None
    result = 0.5 * abs(q / a - c / q)
    return result if math.isfinite(result) and result > 0.0 else None


def mantid_he3_exponent(position, axis, radius, parameters):
    """Calculate Mantid's source-ordered He-3 exponent for a detector.

    ``position`` points from sample to detector in metres; ``axis`` is the
    physical tube axis. ``parameters`` contains pressure (atm), wall thickness
    (m), and temperature (K). Zero denotes an inactive/invalid correction,
    matching the importer's existing exponent contract.
    """
    position, axis = _vector(position), _vector(axis)
    if position is None or axis is None or parameters is None:
        return 0.0
    try:
        pressure, thickness, temperature = (float(item) for item in parameters)
        radius = float(radius)
    except (TypeError, ValueError):
        return 0.0
    if not all(math.isfinite(item) for item in (pressure, thickness, temperature, radius)):
        return 0.0
    straight_path = 2.0 * radius - 2.0 * thickness
    position_length = math.sqrt(_dot(position, position))
    axis_length = math.sqrt(_dot(axis, axis))
    if (pressure <= 0.0 or temperature <= 0.0 or radius <= 0.0
            or straight_path <= 0.0 or position_length <= 0.0 or axis_length <= 0.0):
        return 0.0
    direction = tuple(value / position_length for value in position)
    axis = tuple(value / axis_length for value in axis)
    cosine = _dot(axis, direction)
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    if sine <= 1e-12:
        return 0.0
    pathlength = straight_path / sine
    return HE3_EFFICIENCY_EXPONENTIAL_CONSTANT * (pressure / temperature) * pathlength
