"""Scriptable matrix and orbital-subspace inspection for electronic models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pprint import pformat
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .electronic_structure import ElectronicModel, electronic_model_from_component

ComplexArray = NDArray[np.complex128]


def _readonly_matrix(values: ArrayLike) -> ComplexArray:
    result = np.asarray(values, dtype=np.complex128).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MatrixSubspaceBlock:
    """Norm summary for one row-subspace/column-subspace matrix block."""

    row_subspace: str
    column_subspace: str
    shape: tuple[int, int]
    frobenius_norm: float
    maximum_magnitude: float


@dataclass(frozen=True)
class ElectronicMatrixInspection:
    """One labelled complex matrix with an optional coefficient decomposition."""

    key: str
    title: str
    description: str
    basis_matrix: ComplexArray
    row_labels: tuple[str, ...]
    column_labels: tuple[str, ...]
    row_subspaces: tuple[str, ...]
    column_subspaces: tuple[str, ...]
    coefficient: float = 1.0
    coefficient_label: str = ""
    units: str = ""

    def __post_init__(self) -> None:
        matrix = _readonly_matrix(self.basis_matrix)
        if matrix.shape != (len(self.row_labels), len(self.column_labels)):
            raise ValueError("matrix dimensions must match row and column labels")
        if len(self.row_subspaces) != len(self.row_labels) or len(
            self.column_subspaces
        ) != len(self.column_labels):
            raise ValueError("every matrix row and column requires a subspace label")
        if not np.isfinite(self.coefficient):
            raise ValueError("matrix coefficient must be finite")
        object.__setattr__(self, "basis_matrix", matrix)

    @property
    def contribution_matrix(self) -> ComplexArray:
        """Return coefficient times the displayed matrix basis."""

        result = self.coefficient * self.basis_matrix
        result.setflags(write=False)
        return result

    def subspace_blocks(
        self,
        *,
        contribution: bool = True,
        tolerance: float = 1.0e-12,
    ) -> tuple[MatrixSubspaceBlock, ...]:
        """Summarize nonzero blocks between labelled orbital subspaces."""

        matrix = (
            self.contribution_matrix if contribution else self.basis_matrix
        )
        row_order = tuple(dict.fromkeys(self.row_subspaces))
        column_order = tuple(dict.fromkeys(self.column_subspaces))
        result = []
        for row_name in row_order:
            rows = np.flatnonzero(
                np.asarray(self.row_subspaces, dtype=object) == row_name
            )
            for column_name in column_order:
                columns = np.flatnonzero(
                    np.asarray(self.column_subspaces, dtype=object)
                    == column_name
                )
                block = matrix[np.ix_(rows, columns)]
                norm = float(np.linalg.norm(block))
                if norm <= float(tolerance):
                    continue
                result.append(
                    MatrixSubspaceBlock(
                        row_subspace=row_name,
                        column_subspace=column_name,
                        shape=block.shape,
                        frobenius_norm=norm,
                        maximum_magnitude=float(
                            np.max(np.abs(block), initial=0.0)
                        ),
                    )
                )
        return tuple(result)


def _basis_labels_and_subspaces(
    model: ElectronicModel,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    labels = tuple(state.label for state in model.basis)
    subspaces = tuple(
        "/".join(
            value
            for value in (
                state.site,
                str(state.metadata.get("manifold", "")),
                state.spin,
            )
            if value
        )
        or state.label
        for state in model.basis
    )
    return labels, subspaces


def _builder_term_inspection(
    payload: Mapping[str, Any],
) -> ElectronicMatrixInspection:
    from .electronic_builder import HoppingInvariant, OnsiteInvariant

    if "orbit_label" in payload:
        term = HoppingInvariant.from_dict(payload)
        rows = tuple(term.basis_i)
        columns = tuple(term.basis_j)
        return ElectronicMatrixInspection(
            key=f"builder:{term.identifier}",
            title=term.label,
            description=(
                "Representative symmetry-allowed hopping basis. Rows are "
                "destination orbitals and columns are source orbitals."
            ),
            basis_matrix=term.matrix,
            row_labels=rows,
            column_labels=columns,
            row_subspaces=tuple(label.split(":", 1)[0] for label in rows),
            column_subspaces=tuple(
                label.split(":", 1)[0] for label in columns
            ),
            coefficient=term.value_meV,
            coefficient_label=term.identifier,
            units="meV",
        )
    term = OnsiteInvariant.from_dict(payload)
    labels = tuple(term.basis_labels)
    return ElectronicMatrixInspection(
        key=f"builder:{term.identifier}",
        title=term.label,
        description="Representative symmetry-allowed onsite matrix basis.",
        basis_matrix=term.matrix,
        row_labels=labels,
        column_labels=labels,
        row_subspaces=tuple(label.split(":", 1)[0] for label in labels),
        column_subspaces=tuple(label.split(":", 1)[0] for label in labels),
        coefficient=term.value_meV,
        coefficient_label=term.identifier,
        units="meV",
    )


def electronic_matrix_catalog(
    source: ElectronicModel | Any,
    *,
    reduced_k: Sequence[float] = (0.0, 0.0, 0.0),
    builder_terms: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[ElectronicMatrixInspection, ...]:
    """Return Hamiltonian, term, spin, and builder matrices for inspection."""

    if isinstance(source, ElectronicModel):
        model = source
        configured_terms = tuple(builder_terms or ())
    else:
        model = electronic_model_from_component(source)
        configured_terms = tuple(
            builder_terms
            if builder_terms is not None
            else (
                *source.config.get("onsite_terms", ()),
                *source.config.get("hopping_terms", ()),
            )
        )
    wavevector = np.asarray(reduced_k, dtype=float)
    if wavevector.shape != (3,) or not np.all(np.isfinite(wavevector)):
        raise ValueError("reduced_k must contain three finite coordinates")
    labels, subspaces = _basis_labels_and_subspaces(model)
    result = [
        ElectronicMatrixInspection(
            key="hamiltonian:k",
            title=f"H(k) at {tuple(float(value) for value in wavevector)}",
            description="Resolved Hermitian Bloch Hamiltonian.",
            basis_matrix=model.hamiltonian(wavevector),
            row_labels=labels,
            column_labels=labels,
            row_subspaces=subspaces,
            column_subspaces=subspaces,
            units="meV",
        )
    ]
    phases = np.exp(
        2.0j
        * np.pi
        * np.asarray(model.translations, dtype=float)
        @ wavevector
    )
    for name, blocks in model.parameter_blocks.items():
        selector = np.einsum(
            "r,r,rij->ij",
            phases,
            model.interpolation_weights,
            blocks,
            optimize=True,
        )
        result.append(
            ElectronicMatrixInspection(
                key=f"parameter:{name}",
                title=f"{name} contribution at k",
                description=(
                    "Named Hamiltonian selector after Fourier summation. "
                    "Toggle between its dimensionless basis and coefficient "
                    "contribution in the viewer."
                ),
                basis_matrix=selector,
                row_labels=labels,
                column_labels=labels,
                row_subspaces=subspaces,
                column_subspaces=subspaces,
                coefficient=float(model.parameter_values[name]),
                coefficient_label=name,
                units="meV",
            )
        )
    if model.spin_operators is not None:
        for axis, operator in zip(
            ("x", "y", "z"),
            model.spin_operators,
            strict=True,
        ):
            result.append(
                ElectronicMatrixInspection(
                    key=f"spin:S{axis}",
                    title=f"S{axis}",
                    description="Dimensionless spin-one-half operator.",
                    basis_matrix=operator,
                    row_labels=labels,
                    column_labels=labels,
                    row_subspaces=subspaces,
                    column_subspaces=subspaces,
                )
            )
    result.extend(_builder_term_inspection(item) for item in configured_terms)
    keys = [item.key for item in result]
    if len(keys) != len(set(keys)):
        raise ValueError("matrix-inspection keys must be unique")
    return tuple(result)


def electronic_matrix_script(component: Any) -> str:
    """Return editable Python opening the same matrix-inspection catalog."""

    model = electronic_model_from_component(component)
    builder_terms = (
        *component.config.get("onsite_terms", ()),
        *component.config.get("hopping_terms", ()),
    )
    return "\n".join(
        [
            '"""Inspect electronic Hamiltonian and operator matrices."""',
            "",
            "from nfit import ElectronicModel, electronic_matrix_catalog",
            "from nfit.qt_electronic_matrix_viewer import show_electronic_matrix_catalog",
            "from PySide6 import QtWidgets",
            "",
            f"model = ElectronicModel.from_dict({pformat(model.to_dict(), sort_dicts=True)})",
            f"builder_terms = {pformat(list(builder_terms), sort_dicts=True)}",
            "catalog = electronic_matrix_catalog(model, builder_terms=builder_terms)",
            "app = QtWidgets.QApplication.instance()",
            "owns_app = app is None",
            "if owns_app:",
            "    app = QtWidgets.QApplication([])",
            "window = show_electronic_matrix_catalog(catalog)",
            "if owns_app:",
            "    app.exec()",
            "",
        ]
    )
