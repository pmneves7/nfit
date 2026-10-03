"""Resolve run-dependent IDF locations before detector geometry is cached.

The geometry definition includes referenced run-log values, not just the XML.
Unsupported expressions and location parameters fail before event conversion.
This module has no reducer or GUI dependencies.
"""

from __future__ import annotations

import ast
import hashlib
import math
import xml.etree.ElementTree as ET

import numpy as np

_POSITION_PARAMETERS = {"x", "y", "z", "r-position", "t-position", "p-position"}
_ROTATION_PARAMETERS = {"rotx", "roty", "rotz"}
_LOCATION_PARAMETERS = _POSITION_PARAMETERS | _ROTATION_PARAMETERS


def evaluate_log_expression(expression, value):
    """Evaluate bounded IDF arithmetic using its single numeric ``value``."""
    if not expression:
        return float(value)
    expression = str(expression).replace("^", "**")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"Unsupported IDF logfile expression: {expression!r}") from error
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
               ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Constant, ast.Name, ast.Load)
    if len(list(ast.walk(tree))) > 128 or any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError(f"Unsupported IDF logfile expression: {expression!r}")
    if any(isinstance(node, ast.Name) and node.id != "value" for node in ast.walk(tree)):
        raise ValueError(f"Unsupported IDF logfile expression name: {expression!r}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if type(node.value) not in {int, float}:
                raise ValueError(f"Unsupported IDF logfile expression constant: {expression!r}")
            # Evaluate numerical constants as doubles, including powers, rather
            # than allowing arbitrarily large Python integer intermediates.
            node.value = float(node.value)
    try:
        result = float(eval(compile(tree, "<idf-logfile>", "eval"),
                            {"__builtins__": {}}, {"value": float(value)}))
    except (ArithmeticError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid IDF logfile expression: {expression!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite IDF logfile expression: {expression!r}")
    return result


def numeric_log_values(entry, name, *, fallback_average=False):
    """Read finite raw log values, without an averaging or rounding convention."""
    if not name or entry is None:
        raise ValueError(f"IDF location logfile {name!r} requires referenced run logs")
    group = entry.get("DASlogs/" + name)
    if group is None:
        raise ValueError(f"Referenced instrument log {name!r} is missing")
    dataset = group.get("value")
    if dataset is None and fallback_average:
        dataset = group.get("average_value")
    if dataset is None:
        raise ValueError(f"Referenced instrument log {name!r} has no numeric values")
    values = np.asarray(dataset[()], dtype=float).reshape(-1)
    if not len(values) or np.any(~np.isfinite(values)):
        raise ValueError(f"Referenced instrument log {name!r} is empty or nonfinite")
    return values


def _log_scalar(entry, logfile):
    name = logfile.get("id", "")
    unsupported = set(logfile.attrib) - {"id", "eq", "extract-single-value-as"}
    if unsupported:
        raise ValueError(f"Unsupported IDF logfile attributes: {sorted(unsupported)}")
    mode = logfile.get("extract-single-value-as", "mean")
    position = mode.removeprefix("position") if mode.startswith("position") else None
    if mode not in {"mean", "first", "last", "min", "max", "median"} and not (
        position is not None and position.isdecimal() and int(position) > 0
    ):
        raise ValueError(f"Unsupported IDF logfile extraction: {mode!r}")
    values = numeric_log_values(entry, name, fallback_average=mode == "mean")
    reducers = {"first": lambda a: a[0], "last": lambda a: a[-1],
                "min": np.min, "max": np.max, "median": np.median}
    if mode == "mean":
        scalar = float(values[0])
        if len(values) > 1 and not np.all(values == values[0]):
            times = entry["DASlogs/" + name].get("time")
            if times is None:
                raise ValueError(f"IDF geometry log {name!r} needs timestamps for time-averaged extraction")
            times = np.asarray(times[()], dtype=float).reshape(-1)
            if times.shape != values.shape or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
                raise ValueError(f"IDF geometry log {name!r} needs matching increasing timestamps")
            # Mantid holds each value until the next timestamp, extending the
            # final value by the preceding interval for an unrestricted log.
            durations = np.r_[np.diff(times), times[-1] - times[-2]]
            scalar = float(np.dot(values, durations) / durations.sum())
    elif position is not None:
        index = int(position) - 1
        if index >= len(values):
            raise ValueError(f"IDF logfile extraction {mode!r} exceeds the available values")
        scalar = float(values[index])
    else:
        scalar = float(reducers[mode](values))
    # Mantid serializes the extracted property with C++'s default six
    # significant digits before evaluating the logfile equation.
    scalar = float(format(scalar, ".6g"))
    return evaluate_log_expression(logfile.get("eq"), scalar)


def _parameter_value(entry, parameter):
    logfile = parameter.find("{*}logfile")
    value = parameter.find("{*}value")
    if (logfile is None) == (value is None):
        raise ValueError("IDF location parameter requires exactly one value or logfile")
    if logfile is not None:
        return _log_scalar(entry, logfile)
    try:
        result = float(value.attrib["val"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("IDF location parameter requires a numeric finite value") from error
    if not math.isfinite(result):
        raise ValueError("IDF location parameter requires a numeric finite value")
    return result


def _resolved_location(entry, location):
    if location is None:
        return None
    if not location.findall("{*}parameter"):
        return location
    # Work on a copy: callers can share the original parsed definition.
    resolved = ET.fromstring(ET.tostring(location))
    parameters = {}
    for parameter in list(resolved.findall("{*}parameter")):
        name = parameter.get("name")
        if name not in _LOCATION_PARAMETERS:
            raise ValueError(f"Unsupported IDF location parameter: {name!r}")
        if name in parameters:
            raise ValueError(f"Repeated IDF location parameter: {name!r}")
        parameters[name] = _parameter_value(entry, parameter)
        resolved.remove(parameter)
    for name in ("x", "y", "z"):
        if name in parameters:
            resolved.set(name, format(parameters[name], ".17g"))
    for name, axis in (("r-position", "r"), ("t-position", "t"), ("p-position", "p")):
        if name in parameters:
            resolved.set(axis, format(parameters[name], ".17g"))
    if any(name in parameters for name in _ROTATION_PARAMETERS):
        # Instrument parameters replace the component rotation, rather than
        # adding another Euler rotation to its static location rotation.
        for element in list(resolved.findall("{*}rot")):
            resolved.remove(element)
        for attribute in ("rot", "axis-x", "axis-y", "axis-z"):
            resolved.attrib.pop(attribute, None)
        namespace = resolved.tag.split("}")[0] + "}" if "}" in resolved.tag else ""
        for name, axis in (("rotx", "x"), ("roty", "y"), ("rotz", "z")):
            if name in parameters:
                ET.SubElement(resolved, namespace + "rot", {"val": format(parameters[name], ".17g"), "axis-" + axis: "1"})
    return resolved


def rotation_matrix(angle, axis):
    """Active right-handed Cartesian rotation for an angle in degrees."""
    axis = np.asarray(axis, dtype=float)
    length = np.linalg.norm(axis)
    if not math.isfinite(angle) or np.any(~np.isfinite(axis)) or not length:
        raise ValueError("IDF rotation needs a finite angle and nonzero axis")
    axis = axis / length
    radians = math.radians(angle)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(radians) * cross + (1 - math.cos(radians)) * (cross @ cross)


_rotation = rotation_matrix


def _axis(element):
    explicit = any(element.get("axis-" + key) is not None for key in ("x", "y", "z"))
    return [float(element.get("axis-" + key, 0. if explicit or key != "z" else 1.))
            for key in ("x", "y", "z")]


def location_transform(location, *, entry=None):
    """Cartesian metres and local rotation from a supported IDF location."""
    location = _resolved_location(entry, location)
    if location is None:
        return np.zeros(3), np.eye(3)
    spherical = any(location.get(axis) is not None for axis in ("r", "t", "p"))
    cartesian = any(location.get(axis) is not None for axis in ("x", "y", "z"))
    if spherical and cartesian:
        raise ValueError("IDF location mixes Cartesian and spherical positions")
    if spherical:
        radius, theta, phi = (float(location.get(axis, 0.)) for axis in ("r", "t", "p"))
        theta, phi = math.radians(theta), math.radians(phi)
        translation = radius * np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])
    else:
        translation = np.array([float(location.get(axis, 0.)) for axis in ("x", "y", "z")])
    if np.any(~np.isfinite(translation)):
        raise ValueError("IDF location position must be finite")
    rotation = np.eye(3)
    if location.get("rot") is not None:
        rotation = _rotation(float(location.get("rot")), _axis(location))
    for element in location.findall(".//{*}rot"):
        rotation = rotation @ _rotation(float(element.get("val", 0.)), _axis(element))
    return translation, rotation


def resolved_idf_xml(entry):
    """Return a complete XML cache key with geometry parameters resolved."""
    definition = entry.get("instrument/instrument_xml/data")
    if definition is None:
        raise ValueError("Raw run has no embedded instrument definition")
    xml = definition[()].tobytes()
    root = ET.fromstring(xml)
    locations = root.findall(".//{*}location")
    location_parameters = {id(parameter) for location in locations for parameter in location.findall("{*}parameter")}
    for parameter in root.findall(".//{*}parameter"):
        if (parameter.find("{*}logfile") is not None and parameter.get("name") in (
                _LOCATION_PARAMETERS | {"tube_pressure", "tube_thickness", "tube_temperature"})
                and id(parameter) not in location_parameters):
            raise ValueError("IDF geometry logfile outside a location is unsupported")
    if not location_parameters:
        return xml  # Preserve established static XML cache identity exactly.
    for parent in root.iter():
        for index, child in enumerate(list(parent)):
            if child.tag.rsplit("}", 1)[-1] == "location" and child.findall("{*}parameter"):
                parent.remove(child)
                parent.insert(index, _resolved_location(entry, child))
    return ET.tostring(root)


def resolved_geometry_signature(entry):
    return hashlib.sha256(resolved_idf_xml(entry)).hexdigest()


def source_distance(entry):
    """Resolved moderator-to-origin distance in metres (zero if unspecified)."""
    definition = entry.get("instrument/instrument_xml/data")
    if definition is None:
        return 0.
    root = ET.fromstring(definition[()].tobytes())
    for component in root.findall("{*}component"):
        if component.get("type") == "moderator":
            position, _ = location_transform(component.find("{*}location"), entry=entry)
            return float(np.linalg.norm(position))
    return 0.
