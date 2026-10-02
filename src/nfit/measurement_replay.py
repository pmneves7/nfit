"""Final-grid replay for source-backed event groups without dense covariance."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .measurement_dependencies import SourceReplayRequired
from .project_dataset_io import _json_safe_value

MEASUREMENT_REPLAY_VERSION = 1


def replay_measurement_histogram(group, **binning):
    """Bin original events directly on the requested final HKLE grid.

    This delegates to nfit's public native event reducers and reuses valid
    dataset-owned reduced-event caches. It never redistributes an intermediate
    histogram. Existing numerical policies remain unchanged; same-event copy
    covariance follows the group's explicit symmetry variance policy. Give
    integrated dimensions a single final bin to reunite their contributions.
    Shared calibration uncertainty is not reconstructed by current DGS caches.
    """
    if "raw_dgs" in group.metadata:
        from .raw_dgs import bin_raw_dgs_group
        reducer, source_kind = bin_raw_dgs_group, "raw_dgs"
    elif "mdevent" in group.metadata:
        from .mdevent import bin_mdevent_group
        reducer, source_kind = bin_mdevent_group, "mdevent"
    else:
        raise SourceReplayRequired("Event replay requires the original raw-DGS or MDEvent dataset group")
    selected = list(binning.get("datasets", group.datasets))
    identities = []
    for dataset in selected:
        source = dataset.metadata.get("source_file")
        if not source:
            raise SourceReplayRequired("Replay source identity is missing")
        # Logical MDEvent runs in a shared container are distinct measurements.
        identity = (str(Path(source).expanduser().resolve()), dataset.metadata.get("mdevent_experiment_index"))
        if identity in identities:
            raise SourceReplayRequired("Repeated source measurements require explicit shared-source lineage")
        identities.append(identity)
    output = reducer(group, **binning)
    configuration = copy.deepcopy(group.metadata[source_kind])
    serializable_binning = {key: value for key,value in binning.items()
                           if key not in {"datasets", "progress_callback"}}
    signature = hashlib.sha256(json.dumps(_json_safe_value({"source": configuration,
        "identities": identities, "binning": serializable_binning}), sort_keys=True).encode()).hexdigest()
    metadata = dict(output.metadata)
    metadata["measurement_replay"] = {"version": MEASUREMENT_REPLAY_VERSION,
        "source_kind": source_kind, "source_identities": identities,
        "recipe_signature": signature, "binning": _json_safe_value(serializable_binning),
        "source_configuration": _json_safe_value(configuration),
        "uncertainty": "final_bin_event_variance_under_recorded_copy_policy",
        "normalizer": "known", "shared_calibration_uncertainty": "not_represented"}
    return output.with_updates(metadata=metadata)
