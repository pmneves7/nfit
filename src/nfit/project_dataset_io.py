"""Compatibility imports for portable dataset archives and project saving.

The codec lives in :mod:`nfit.dataset_archive`. Saving a project dataset selects
its original payload or prepared view in :mod:`nfit.project_data`; that public
function is resolved lazily here so importing codec compatibility helpers does
not load the preparation facade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .dataset_archive import (
    DATASET_CONTEXT_KEYS as DATASET_CONTEXT_KEYS,
)
from .dataset_archive import (
    KINEMATIC_KF_KI_INCLUDED_KEY as KINEMATIC_KF_KI_INCLUDED_KEY,
)
from .dataset_archive import (
    _decode_nfit_dataset_file as _decode_nfit_dataset_file,
)
from .dataset_archive import (
    _json_safe_value as _json_safe_value,
)
from .dataset_archive import (
    _load_nfit_dataset_file as _load_nfit_dataset_file,
)
from .dataset_archive import (
    _load_nfit_mdhisto_archive as _load_nfit_mdhisto_archive,
)
from .dataset_archive import (
    _load_nfit_point_list_archive as _load_nfit_point_list_archive,
)
from .dataset_archive import (
    _nfit_archive_json_mapping as _nfit_archive_json_mapping,
)
from .dataset_archive import (
    _nfit_archive_optional_int as _nfit_archive_optional_int,
)
from .dataset_archive import (
    _nfit_archive_text as _nfit_archive_text,
)
from .dataset_archive import (
    _save_point_list_file as _save_point_list_file,
)

if TYPE_CHECKING:
    from .project_data import save_dataset_file as save_dataset_file

__all__ = ["DATASET_CONTEXT_KEYS", "KINEMATIC_KF_KI_INCLUDED_KEY", "save_dataset_file"]


def __getattr__(name: str):
    if name == "save_dataset_file":
        from .project_data import save_dataset_file

        return save_dataset_file
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"save_dataset_file"})
