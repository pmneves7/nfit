from __future__ import annotations

import numpy as np
import pytest

import nfit.electronic_pipeline_sampling as pipeline_sampling
from nfit import (
    DataGroup,
    DatasetEntry,
    FitDatasetInput,
    ModelComponentSpec,
    PointData4D,
    SamplingCertificate,
    build_electronic_model,
    certify_electronic_pipeline_sampling,
    certify_group_lindhard_sampling,
    certify_lindhard_pipeline_sampling,
    electronic_pipeline_components,
    electronic_pipeline_state_digest,
    model_definition,
    sample_pipeline_datasets,
    sampling_policy,
)


def _defaults(model_type: str) -> dict:
    return {
        field.name: field.default
        for field in model_definition(model_type).config_fields
    }


def _pipeline_components(*, hopping_meV: float = 0.0):
    model = build_electronic_model(
        direct_lattice=np.diag([2.0, 8.0, 9.0]),
        basis=["s"],
        hoppings={(1, 0, 0): [[hopping_meV]]},
        periodic_axes=(0,),
        energy_unit="meV",
    )
    tight_binding = ModelComponentSpec(
        name="bands",
        type="tight_binding",
        config={
            **_defaults("tight_binding"),
            "model_data": model.to_dict(),
            "periodic_axes": [0],
        },
    )
    lindhard = ModelComponentSpec(
        name="bare",
        type="lindhard",
        parameters={"broadening": 5.0},
        config={
            **_defaults("lindhard"),
            "electronic_component": "bands",
            "response_mesh": [12],
            "response_mesh_shift": [0.0],
            "response_sampling_accuracy": "preview",
            "response_sampling_max_refinements": 4,
            "response_sampling_max_mesh_points": 64,
            "response_sampling_points_per_dataset": 4,
            "formula_units_mode": "manual",
            "formula_units_per_cell": 1.0,
            "magnetic_normalization_mode": "manual",
            "magnetic_centers_per_model_cell": 1.0,
            # Certification must override these interpolation settings in its
            # private component copies without changing the project setting.
            "response_q_evaluation": "interpolated",
            "response_q_interpolation_rtol": 0.2,
            "response_q_interpolation_mesh": [4],
        },
    )
    stoner = ModelComponentSpec(
        name="dressed",
        type="stoner_rpa",
        parameters={"I": 0.01},
        config={
            **_defaults("stoner_rpa"),
            "response_component": "bare",
        },
    )
    return tight_binding, lindhard, stoner


def _dataset(name: str = "scan", *, size: int = 9, data_type: str = "single_crystal_inelastic"):
    points = PointData4D(
        H=np.linspace(0.1, 0.9, size),
        K=np.zeros(size),
        L=np.zeros(size),
        E=np.linspace(1.0, 4.0, size),
        intensity=np.zeros(size),
        sigma=np.ones(size),
        temperature=np.linspace(10.0, 30.0, size),
        magnetic_field=np.column_stack(
            (np.zeros(size), np.zeros(size), np.linspace(0.0, 2.0, size))
        ),
        metadata={"normalization": "test"},
    )
    return FitDatasetInput(name, points, data_type=data_type)


def test_representative_dataset_sampling_is_deterministic_and_domain_sensitive():
    dataset = _dataset(size=101)

    first, first_domain = sample_pipeline_datasets(
        [dataset], max_points_per_dataset=8
    )
    second, second_domain = sample_pipeline_datasets(
        [dataset], max_points_per_dataset=8
    )

    assert first_domain == second_domain
    assert first_domain[0].sampled_points == 8
    assert first_domain[0].sampled_indices[0] == 0
    assert first_domain[0].sampled_indices[-1] == 100
    assert first_domain[0].coordinate_ranges["temperature_K"] == (10.0, 30.0)
    assert first_domain[0].coordinate_ranges["field_z_T"] == (0.0, 2.0)
    np.testing.assert_array_equal(first[0].data.H, second[0].data.H)

    changed_points = dataset.data.with_updates(
        mask=np.arange(dataset.data.size) % 2 == 0
    )
    _changed, changed_domain = sample_pipeline_datasets(
        [
            FitDatasetInput(
                dataset.name,
                changed_points,
                data_type=dataset.data_type,
            )
        ],
        max_points_per_dataset=8,
    )
    assert changed_domain[0].domain_digest != first_domain[0].domain_digest


def test_representative_sampling_preserves_independent_coordinate_extrema():
    coordinates = np.zeros((20, 4), dtype=float)
    for axis in range(4):
        coordinates[2 * axis, axis] = -1.0
        coordinates[2 * axis + 1, axis] = 1.0
    points = PointData4D(
        H=coordinates[:, 0],
        K=coordinates[:, 1],
        L=coordinates[:, 2],
        E=coordinates[:, 3],
        intensity=np.zeros(20),
        sigma=np.ones(20),
        temperature=10.0,
    )

    _sampled, domains = sample_pipeline_datasets(
        [FitDatasetInput("independent", points)],
        max_points_per_dataset=8,
    )

    assert set(domains[0].sampled_indices) == set(range(8))


def test_pipeline_closure_and_digest_include_rpa_state_but_not_saved_certificate():
    tight_binding, lindhard, stoner = _pipeline_components()
    components = {
        item.name: item for item in (tight_binding, lindhard, stoner)
    }

    closure = electronic_pipeline_components(lindhard, components)
    assert [(item.name, item.type) for item in closure] == [
        ("bands", "tight_binding"),
        ("bare", "lindhard"),
        ("dressed", "stoner_rpa"),
    ]
    original = electronic_pipeline_state_digest(lindhard, components)
    lindhard.config["response_sampling_certificate"] = {"status": "test"}
    assert electronic_pipeline_state_digest(lindhard, components) == original
    stoner.parameters["I"] = 0.02
    assert electronic_pipeline_state_digest(lindhard, components) != original


def test_pipeline_sampling_resolves_one_automatic_tight_binding_source():
    tight_binding, lindhard, stoner = _pipeline_components()
    lindhard.config["electronic_component"] = ""
    components = {
        item.name: item for item in (tight_binding, lindhard, stoner)
    }

    certificate = certify_electronic_pipeline_sampling(
        lindhard,
        components,
        [_dataset()],
        policy=sampling_policy("preview"),
        max_refinements=4,
        max_mesh_points=64,
        max_points_per_dataset=4,
    )

    assert certificate.certified
    assert certificate.provenance["source_component"] == "bands"
    assert lindhard.config["electronic_component"] == ""


def test_complete_pipeline_certificate_uses_rpa_and_exact_q_copies(monkeypatch):
    tight_binding, lindhard, stoner = _pipeline_components()
    components = {
        item.name: item for item in (tight_binding, lindhard, stoner)
    }
    observed_q_settings = []
    original_compile = pipeline_sampling.compile_fit_problem

    def inspected_compile(configured, datasets, **kwargs):
        selected = next(item for item in configured if item.name == "bare")
        observed_q_settings.append(
            (
                selected.config["response_q_evaluation"],
                selected.config["response_q_interpolation_rtol"],
                selected.config["response_q_interpolation_atol"],
            )
        )
        return original_compile(configured, datasets, **kwargs)

    monkeypatch.setattr(pipeline_sampling, "compile_fit_problem", inspected_compile)
    certificate = certify_electronic_pipeline_sampling(
        lindhard,
        components,
        [_dataset()],
        policy=sampling_policy("preview"),
        max_refinements=4,
        max_mesh_points=64,
        max_points_per_dataset=4,
    )

    assert certificate.certified
    assert certificate.observable == "electronic_pipeline"
    assert certificate.domain["kind"] == "fit_datasets"
    assert certificate.domain["dataset_count"] == 1
    assert certificate.provenance["q_evaluation_during_certificate"] == "direct"
    assert certificate.provenance["pipeline_components"][-1] == {
        "name": "dressed",
        "type": "stoner_rpa",
    }
    assert observed_q_settings
    assert set(observed_q_settings) == {("direct", 0.0, 0.0)}
    assert lindhard.config["response_q_evaluation"] == "interpolated"
    restored = SamplingCertificate.from_dict(certificate.to_dict())
    assert restored == certificate


def test_pipeline_aggregation_requires_every_applicable_dataset(monkeypatch):
    tight_binding, lindhard, stoner = _pipeline_components()
    components = {
        item.name: item for item in (tight_binding, lindhard, stoner)
    }

    def predictions(_closure, datasets, *, lindhard_name, mesh):
        del lindhard_name
        # One observable is perfectly stable.  The other changes by much more
        # than the 5% preview tolerance, so aggregation must not average it away.
        return {
            dataset.name: (
                np.ones(dataset.data.size)
                if dataset.name == "stable"
                else np.full(dataset.data.size, float(mesh[0]))
            )
            for dataset in datasets
        }

    monkeypatch.setattr(pipeline_sampling, "_compiled_predictions", predictions)
    certificate = certify_electronic_pipeline_sampling(
        lindhard,
        components,
        [_dataset("stable"), _dataset("unstable")],
        policy=sampling_policy("preview"),
        max_refinements=4,
        max_mesh_points=64,
        max_points_per_dataset=3,
    )

    assert certificate.status == "budget_exhausted"
    assert all(not comparison.passed for comparison in certificate.comparisons)
    assert certificate.provenance["stopping_reason"] == (
        "insufficient_remaining_refinements"
    )
    details = certificate.provenance["dataset_comparisons"][0]["datasets"]
    assert details["stable"]["passed"]
    assert not details["unstable"]["passed"]


def test_unrelated_dataset_types_are_ignored_without_weakening_domain():
    tight_binding, lindhard, stoner = _pipeline_components()
    components = {
        item.name: item for item in (tight_binding, lindhard, stoner)
    }
    certificate = certify_electronic_pipeline_sampling(
        lindhard,
        components,
        [_dataset("scan"), _dataset("calorimetry", data_type="heat_capacity")],
        policy=sampling_policy("preview"),
        max_refinements=4,
        max_mesh_points=64,
        max_points_per_dataset=3,
    )

    assert certificate.domain["dataset_count"] == 1
    assert certificate.domain["datasets"][0]["name"] == "scan"


def test_pipeline_domain_supports_powder_and_bulk_dataset_coordinates():
    tight_binding, lindhard, stoner = _pipeline_components()
    components = {
        item.name: item for item in (tight_binding, lindhard, stoner)
    }
    powder = _dataset("powder", size=5, data_type="powder_inelastic")
    powder = FitDatasetInput(
        powder.name,
        powder.data.with_updates(
            metadata={
                **powder.data.metadata,
                "coordinate_units": "1/angstrom",
                "powder_q_modulus_axis": True,
            }
        ),
        data_type=powder.data_type,
    )
    bulk_points = PointData4D(
        H=np.zeros(5),
        K=np.zeros(5),
        L=np.zeros(5),
        E=np.zeros(5),
        intensity=np.zeros(5),
        sigma=np.ones(5),
        temperature=np.linspace(5.0, 25.0, 5),
        magnetic_field=np.column_stack(
            (np.zeros(5), np.zeros(5), np.linspace(0.1, 0.5, 5))
        ),
    )
    bulk = FitDatasetInput(
        "magnetization",
        bulk_points,
        data_type="magnetization",
    )

    certificate = certify_electronic_pipeline_sampling(
        lindhard,
        components,
        [powder, bulk],
        policy=sampling_policy("preview"),
        max_refinements=4,
        max_mesh_points=64,
        max_points_per_dataset=4,
    )

    assert certificate.certified
    assert {item["data_type"] for item in certificate.domain["datasets"]} == {
        "powder_inelastic",
        "magnetization",
    }
    bulk_domain = next(
        item
        for item in certificate.domain["datasets"]
        if item["name"] == "magnetization"
    )
    assert bulk_domain["coordinate_ranges"]["temperature_K"] == [5.0, 25.0]
    assert bulk_domain["coordinate_ranges"]["field_z_T"] == [0.1, 0.5]


def test_component_wrapper_updates_mesh_only_after_successful_certificate():
    tight_binding, lindhard, stoner = _pipeline_components()
    components = {
        item.name: item for item in (tight_binding, lindhard, stoner)
    }
    certificate = certify_lindhard_pipeline_sampling(
        lindhard,
        components,
        [_dataset()],
    )
    assert certificate.certified
    assert lindhard.config["response_mesh"] == list(certificate.chosen_mesh)
    assert lindhard.config["response_sampling_certificate"]["observable"] == (
        "electronic_pipeline"
    )

    _tight_binding, limited, limited_stoner = _pipeline_components()
    limited.config["response_sampling_max_mesh_points"] = 1
    limited_components = {
        item.name: item
        for item in (_tight_binding, limited, limited_stoner)
    }
    original_mesh = list(limited.config["response_mesh"])
    exhausted = certify_lindhard_pipeline_sampling(
        limited,
        limited_components,
        [_dataset()],
    )
    assert exhausted.status == "budget_exhausted"
    assert limited.config["response_mesh"] == original_mesh
    assert limited.config["response_sampling_certificate"]["status"] == (
        "budget_exhausted"
    )


def test_group_wrapper_uses_the_ordinary_fit_dataset_preparation_path():
    tight_binding, lindhard, stoner = _pipeline_components()
    source = _dataset()
    group = DataGroup(
        name="workspace",
        datasets=[
            DatasetEntry(
                name=source.name,
                data=source.data,
                kind="test",
                data_type=source.data_type,
            )
        ],
        models={
            item.name: item for item in (tight_binding, lindhard, stoner)
        },
    )

    certificate = certify_group_lindhard_sampling(group, lindhard)

    assert certificate.certified
    assert certificate.domain["datasets"][0]["name"] == "scan"
    assert certificate.domain["datasets"][0]["coordinate_ranges"][
        "temperature_K"
    ] == [10.0, 30.0]


def test_pipeline_sampling_rejects_a_group_without_applicable_datasets():
    tight_binding, lindhard, stoner = _pipeline_components()
    components = {
        item.name: item for item in (tight_binding, lindhard, stoner)
    }
    with pytest.raises(ValueError, match="no dataset is matched|no enabled prepared"):
        certify_electronic_pipeline_sampling(
            lindhard,
            components,
            [_dataset("calorimetry", data_type="heat_capacity")],
            policy=sampling_policy("preview"),
            max_refinements=4,
            max_mesh_points=64,
        )
