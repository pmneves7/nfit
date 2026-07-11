from __future__ import annotations

from typing import Literal


AxisRole = Literal[
    "h",
    "k",
    "l",
    "q_modulus",
    "momentum_projection",
    "energy_transfer",
    "intensity",
    "uncertainty",
    "unknown",
]


def infer_axis_role(
    name: str,
    *,
    units: str = "",
    kind: str | None = None,
) -> AxisRole:
    """Infer a reduced-data axis role from file metadata.

    This helper is intentionally instrument-agnostic. Import adapters should
    prefer explicit labels supplied by the file, then fall back to units and
    coordinate kind. The returned role is descriptive rather than prescriptive:
    projected momentum axes such as ``[H,H,0]`` remain projections, while
    scalar ``|Q|`` axes are tagged separately from reciprocal-lattice ``H``,
    ``K``, and ``L`` axes.
    """

    text = _normalized_axis_text(name)
    unit_text = _normalized_axis_text(units)
    kind_text = _normalized_axis_text("" if kind is None else kind)
    combined = f"{text} {unit_text} {kind_text}"

    if text in {"h", "rluh", "hindex"}:
        return "h"
    if text in {"k", "rluk", "kindex"}:
        return "k"
    if text in {"l", "rlul", "lindex"}:
        return "l"
    if text in {"i", "intensity", "signal", "counts", "scatteringintensity"}:
        return "intensity"
    if text in {"error", "errors", "sigma", "uncertainty", "stderr", "err"}:
        return "uncertainty"
    if _looks_like_q_modulus(text, unit_text):
        return "q_modulus"
    if _looks_like_energy_transfer(combined):
        return "energy_transfer"
    if _looks_like_rlu_projection(name, combined):
        return "momentum_projection"
    return "unknown"


def _looks_like_energy_transfer(text: str) -> bool:
    return (
        "deltae" in text
        or "energytransfer" in text
        or text in {"e", "energy", "omega"}
        or "mev" in text
    )


def _looks_like_q_modulus(text: str, unit_text: str) -> bool:
    q_names = {
        "q",
        "modq",
        "qmod",
        "qmag",
        "qabs",
        "qmagnitude",
        "momentumtransfer",
    }
    return (
        text in q_names
        or "q" in text and "angstrom" in unit_text
        or "inverseangstrom" in unit_text and text in {"", "momentum"}
    )


def _looks_like_rlu_projection(original_name: str, text: str) -> bool:
    bracket_projection = "[" in original_name and "]" in original_name
    return bracket_projection or "rlu" in text or "hkl" in text


def _normalized_axis_text(text: str) -> str:
    normalized = str(text).strip().lower()
    normalized = normalized.replace("å", "angstrom").replace("Å", "angstrom")
    normalized = normalized.replace("a^-1", "inverseangstrom")
    normalized = normalized.replace("å^-1", "inverseangstrom")
    normalized = normalized.replace("angstrom^-1", "inverseangstrom")
    normalized = normalized.replace("1/angstrom", "inverseangstrom")
    normalized = normalized.replace("inverse angstrom", "inverseangstrom")
    replacements = {
        "|": "",
        "(": "",
        ")": "",
        "[": "",
        "]": "",
        ",": "",
        "/": "",
        "\\": "",
        "^": "",
        "-": "",
        "_": "",
        " ": "",
        ".": "",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized
