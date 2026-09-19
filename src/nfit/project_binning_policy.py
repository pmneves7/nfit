"""Pure ownership and presentation policy for project rebin recipes.

The functions in this module inspect project configuration only.  They do not
create recipes, resolve numerical grids, or load dataset payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .pipeline import BackgroundSpec, DataGroup, DatasetEntry, DatasetGroup
from .project_composites import GROUP_COMPOSITE_BINNINGS_KEY
from .project_imports import GROUP_COMPOSITE_KEY
from .project_rebinning import DATASET_REBIN_BINNINGS_KEY

RebinOwner = DataGroup | DatasetGroup | DatasetEntry


@dataclass(frozen=True)
class RebinPresentationPolicy:
    """Describe which project object owns the rebin UI for a selection."""

    selection: RebinOwner
    owner: RebinOwner | None
    owner_kind: Literal["dataset", "composite", "none"]
    show_editor: bool
    show_binning_selector: bool
    linked: bool
    allow_independent_recipe: bool
    message: str


def _composite_enabled(owner: DataGroup | DatasetGroup) -> bool:
    config = owner.metadata.get(GROUP_COMPOSITE_KEY)
    return bool(isinstance(config, dict) and config.get("enabled"))


def _has_enabled_named_binning(owner: RebinOwner) -> bool:
    if isinstance(owner, DatasetEntry):
        registry = owner.parameters.get(DATASET_REBIN_BINNINGS_KEY)
    else:
        registry = owner.metadata.get(GROUP_COMPOSITE_BINNINGS_KEY)
    if not isinstance(registry, dict) or not isinstance(registry.get("items"), list):
        return False
    return any(
        isinstance(item, dict)
        and isinstance(item.get("config"), dict)
        and bool(item["config"].get("enabled"))
        for item in registry["items"]
    )


def _paths(
    root: DataGroup,
) -> tuple[
    dict[int, tuple[DataGroup | DatasetGroup, ...]],
    dict[int, tuple[DataGroup | DatasetGroup, ...]],
]:
    """Return identity-keyed ancestor paths without retaining numerical data."""

    groups: dict[int, tuple[DataGroup | DatasetGroup, ...]] = {id(root): ()}
    datasets: dict[int, tuple[DataGroup | DatasetGroup, ...]] = {}

    def visit(
        node: DataGroup | DatasetGroup,
        ancestors: tuple[DataGroup | DatasetGroup, ...],
    ) -> None:
        path = (*ancestors, node)
        for dataset in node.datasets:
            datasets[id(dataset)] = path
        for subgroup in node.subgroups:
            groups[id(subgroup)] = path
            visit(subgroup, path)

    visit(root, ())
    return groups, datasets


def _native_event_collection(
    dataset: DatasetEntry,
    ancestors: tuple[DataGroup | DatasetGroup, ...],
) -> DataGroup | DatasetGroup | None:
    if dataset.kind not in {"mdevent", "raw_dgs_nexus"}:
        return None
    metadata_key = "mdevent" if dataset.kind == "mdevent" else "raw_dgs"
    for owner in reversed(ancestors):
        if isinstance(owner.metadata.get(metadata_key), dict):
            return owner
    # Imported event leaves are histogrammed as a collection even when older
    # projects do not carry the newer collection metadata marker.
    return ancestors[-1] if ancestors else None


def rebin_presentation_policy(
    root: DataGroup,
    selection: RebinOwner,
) -> RebinPresentationPolicy:
    """Return the effective rebin owner and UI state for ``selection``.

    The nearest enabled composite wins for nested collections.  A native event
    leaf is always controlled by its histogram-producing collection.  Ordinary
    member datasets keep their private recipes available as a secondary action;
    the policy never mutates or removes those recipes.
    """

    group_paths, dataset_paths = _paths(root)
    if isinstance(selection, DatasetEntry):
        ancestors = dataset_paths.get(id(selection))
        if ancestors is None:
            raise ValueError("selected dataset is not contained by the root data group")
        native_owner = _native_event_collection(selection, ancestors)
        enabled_owner = next(
            (owner for owner in reversed(ancestors) if _composite_enabled(owner)),
            None,
        )
        owner = native_owner or enabled_owner
        if owner is not None:
            native = native_owner is not None
            reason = (
                "This native event source is histogrammed on its collection's grid."
                if native
                else "This dataset is a member of an enabled composite and uses that collection's grid."
            )
            return RebinPresentationPolicy(
                selection=selection,
                owner=owner,
                owner_kind="composite",
                show_editor=False,
                show_binning_selector=_has_enabled_named_binning(selection),
                linked=True,
                allow_independent_recipe=not native,
                message=reason,
            )
        return RebinPresentationPolicy(
            selection=selection,
            owner=selection,
            owner_kind="dataset",
            show_editor=True,
            show_binning_selector=True,
            linked=False,
            allow_independent_recipe=False,
            message="This independent dataset owns its rebin recipe.",
        )

    if selection is not root and id(selection) not in group_paths:
        raise ValueError("selected collection is not contained by the root data group")
    if _composite_enabled(selection):
        return RebinPresentationPolicy(
            selection=selection,
            owner=selection,
            owner_kind="composite",
            show_editor=True,
            show_binning_selector=True,
            linked=False,
            allow_independent_recipe=False,
            message="This enabled collection owns the composite output grid.",
        )
    named_output = _has_enabled_named_binning(selection)
    return RebinPresentationPolicy(
        selection=selection,
        owner=selection if named_output else None,
        owner_kind="composite" if named_output else "none",
        show_editor=False,
        show_binning_selector=named_output,
        linked=False,
        allow_independent_recipe=False,
        message="Enable combining for this collection to define a composite output grid.",
    )


def background_rebin_explanation(
    background: BackgroundSpec,
    *,
    owner: DataGroup | DatasetGroup | DatasetEntry | None = None,
) -> str:
    """Explain whether a linked background's private rebin recipe is consumed."""

    if not background.enabled:
        return "This background link is disabled, so no background rebin recipe is consumed."
    if background.projection == "measured_events":
        return (
            "Measured-event subtraction replays the source events on the sample grid; "
            "the background source's private view recipe is not used. Source masks and "
            "the configured background scale still apply."
        )
    if isinstance(owner, DatasetEntry):
        return (
            "Dataset background subtraction evaluates the linked dataset's enabled private "
            "rebin recipe before subtraction (including any required interpolation)."
        )
    if background.source_group is not None or bool(background.source_group_id):
        return (
            "This collection background uses the linked collection's composite rebin recipe "
            "before center or sample-trajectory projection."
        )
    if background.source_entry is not None or bool(background.source_dataset_id):
        return (
            "For a collection background, raw point data follow the collection output grid, "
            "while histogram data must already match or use the configured powder projection; "
            "the source's private view recipe does not control subtraction."
        )
    return (
        "The background source is unresolved. Once linked, collection sources use their "
        "composite recipe for powder projection; measured-event replay ignores view recipes."
    )


def background_source_explanations(
    root: DataGroup,
    selection: DataGroup | DatasetGroup | DatasetEntry,
) -> tuple[str, ...]:
    """Describe enabled background consumers of ``selection`` by stable identity or ID."""

    selected_id = getattr(selection, "id", None)
    messages: list[str] = []

    def references(background: BackgroundSpec) -> bool:
        if isinstance(selection, DatasetEntry):
            return background.source_entry is selection or (
                bool(selected_id) and background.source_dataset_id == selected_id
            )
        return background.source_group is selection or (
            bool(selected_id) and background.source_group_id == selected_id
        )

    def inspect(owner: DataGroup | DatasetGroup | DatasetEntry) -> None:
        for background in owner.backgrounds:
            if references(background):
                messages.append(
                    f"{'Used by' if background.enabled else 'Linked but disabled for'} "
                    f"{owner.name!r}: "
                    f"{background_rebin_explanation(background, owner=owner)}"
                )

    inspect(root)
    for dataset in root.datasets:
        inspect(dataset)
    for subgroup in root.iter_subgroups():
        inspect(subgroup)
        for dataset in subgroup.datasets:
            inspect(dataset)
    return tuple(messages)
