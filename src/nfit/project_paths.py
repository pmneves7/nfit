"""Portable declared project paths and metadata-only relocation.

Only schema-owned path fields and machine-generated cache identities are
rewritten. A project directory is an anchor, not a filesystem search root.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from pathlib import Path
from threading import RLock

PORTABLE_PATHS_KEY = "paths"
_FILE_FIELDS = frozenset((
    "source_file", "source_project", "source_path", "normalization_file",
    "mask_file", "flux_file", "ub_file", "calibration_file",
    "normalization_path", "mask_path", "flux_path", "ub_path",
))
_FILE_LIST_FIELDS = frozenset(("source_files", "resolved_files"))
_IDENTITY_FIELDS = frozenset(("source_identity", "source_namespace"))


def _absolute(value, directory):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = directory / path
    try:
        return path.resolve()
    except OSError:
        # A foreign home or disconnected mount can deny traversal even with
        # strict=False. Keep its lexical identity available for relocation.
        return Path(os.path.abspath(path))


def _exists(path):
    try:
        return Path(path).exists()
    except OSError:
        return False


def _field_kind(key, parents, owner):
    if key in _FILE_FIELDS:
        if key == "source_file" and owner.get(key) in (
            owner.get("analysis_artifact_path"), owner.get("project_artifact_path")
        ):
            return None
        return "file"
    if key in _FILE_LIST_FIELDS:
        return "files"
    if key == "source_ids" and "source_lineage" in parents:
        return "identities"
    if key in _IDENTITY_FIELDS and isinstance(owner.get(key), str):
        return "identity"
    if key == "directory" and "source_selection" in parents:
        return "directory"
    if key == "temporary_storage_directory":
        return "directory"
    if key == "path" and (
        "source_identity" in parents or "source_selection" in parents
        or {"size_bytes", "mtime_ns", "available"}.intersection(owner)
    ):
        return "file"
    return None


def _walk(value, transform, *, parents=()):
    """Copy JSON containers while retaining opaque strings and array payloads."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if key == PORTABLE_PATHS_KEY and not parents:
                result[key] = item
                continue
            kind = _field_kind(key, parents, value)
            if kind in ("file", "directory", "identity") and isinstance(item, str) and item:
                result[key] = transform(item, kind)
            elif kind in ("files", "identities") and isinstance(item, (tuple, list)):
                result[key] = [transform(entry, "identity" if kind == "identities" else "file") if isinstance(entry, str) and entry
                               else _walk(entry, transform, parents=(*parents, key)) for entry in item]
            else:
                result[key] = _walk(item, transform, parents=(*parents, key))
        return result
    if isinstance(value, list):
        return [_walk(item, transform, parents=parents) for item in value]
    if isinstance(value, tuple):
        return tuple(_walk(item, transform, parents=parents) for item in value)
    return value


def _identity_parts(value):
    path, separator, suffix = value.partition("#")
    return path, separator + suffix if separator else ""


def _declared_paths(value, *, parents=()):
    """Visit declared references without copying the manifest or opening data."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == PORTABLE_PATHS_KEY and not parents:
                continue
            kind = _field_kind(key, parents, value)
            if kind in ("file", "directory", "identity") and isinstance(item, str) and item:
                yield item, kind
            elif kind in ("files", "identities") and isinstance(item, (tuple, list)):
                for entry in item:
                    if isinstance(entry, str) and entry:
                        yield entry, "identity" if kind == "identities" else "file"
                    else:
                        yield from _declared_paths(entry, parents=(*parents, key))
            else:
                yield from _declared_paths(item, parents=(*parents, key))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _declared_paths(item, parents=parents)


class ProjectPathResolution:
    """Small runtime relocation map; no scientific payloads are retained."""

    def __init__(self, project_path, *, recorded=None, path_mappings=None):
        self._lock = RLock()
        self.directory = _absolute(Path(project_path).expanduser().absolute().parent, Path.cwd())
        recorded = recorded if isinstance(recorded, Mapping) and recorded.get("version") == 1 else {}
        self.origin = Path(recorded["origin"]) if recorded.get("origin") else None
        self.originals = dict(recorded.get("originals", {}))
        self.aliases = {}
        self.roots = []
        self._resolved = {}
        for old, new in (path_mappings or {}).items():
            old = str(_absolute(old, self.origin or self.directory))
            self.roots.append((old, str(_absolute(new, self.directory))))
        for old, new in recorded.get("root_mappings", {}).items():
            self.roots.append((str(old), str(_absolute(new, self.directory))))
        for old, new in recorded.get("aliases", {}).items():
            self.aliases[str(old)] = str(_absolute(new, self.directory))

    def __deepcopy__(self, memo):
        # Runtime path memoization is shared; it owns no project or data arrays.
        memo[id(self)] = self
        return self

    def _mapped(self, path):
        value = Path(path)
        for old, new in sorted(self.roots, key=lambda pair: len(Path(pair[0]).parts), reverse=True):
            try:
                suffix = value.relative_to(old)
            except ValueError:
                continue
            return str(_absolute(Path(new) / suffix, self.directory))
        return None

    def _legacy_candidate(self, path):
        """Require a shared multi-directory suffix and an existing exact target."""
        candidates = {}
        old_path = Path(path)
        for old in old_path.parents:
            for new in (self.directory, *self.directory.parents):
                shared = 0
                for first, second in zip(reversed(old.parts), reversed(new.parts), strict=False):
                    if first != second or first in (os.sep, ""):
                        break
                    shared += 1
                if shared < 2:
                    continue
                target = new / old_path.relative_to(old)
                if _exists(target):
                    canonical = str(_absolute(target, self.directory))
                    previous = candidates.get(canonical)
                    if previous is None or shared > previous[0]:
                        candidates[canonical] = (shared, str(old), str(new))
        if not candidates:
            return None
        score = max(record[0] for record in candidates.values())
        winners = [(target, record) for target, record in candidates.items() if record[0] == score]
        if len(winners) != 1:
            return None
        target, (shared, old, new) = winners[0]
        # A proven matching suffix anchors its common experiment root too, so
        # sibling nexus/calibration directories resolve regardless of field order.
        old_root = Path(old).parents[shared - 2]
        new_root = Path(new).parents[shared - 2]
        self.roots.append((str(old_root), str(new_root)))
        return target

    def prime(self, payload):
        """Discover legacy roots before applying any field or cache identity."""
        for path, kind in _declared_paths(payload):
            self.resolve(path, kind)
        for path, result in tuple(self._resolved.items()):
            original = self.originals.get(path, path)
            mapped = self._mapped(original)
            if mapped is not None and mapped != result:
                del self._resolved[path]

    def resolve(self, value, kind="file"):
        with self._lock:
            return self._resolve(value, kind)

    def _resolve(self, value, kind):
        path, suffix = _identity_parts(value) if kind == "identity" else (value, "")
        if kind == "identity" and not Path(path).is_absolute() and path not in self.originals and path not in self.aliases:
            return value
        if path in self._resolved:
            return self._resolved[path] + suffix
        original = self.originals.get(path)
        lexical = str(_absolute(path, self.directory)) if not Path(path).expanduser().is_absolute() else str(Path(path).expanduser())
        result = self._mapped(original or lexical)
        if result is None and lexical in self.aliases:
            result = self.aliases[lexical]
        if result is None:
            candidate = Path(lexical)
            if _exists(candidate):
                result = str(_absolute(candidate, self.directory))
            elif original and _exists(original):
                result = str(_absolute(original, self.directory))
            else:
                result = self._legacy_candidate(original or lexical) or lexical
        self._resolved[path] = result
        self.aliases[path] = result
        if original:
            self.aliases[str(original)] = result
        self.aliases[lexical] = result
        return result + suffix

    def _machine_string(self, value):
        """Translate exact declared paths inside generated signature payloads."""
        if value in self.aliases:
            return self.aliases[value]
        if value.startswith(("[", "{")):
            try:
                decoded = json.loads(value)
            except (TypeError, ValueError):
                return value
            updated = self.machine_value(decoded)
            return json.dumps(updated, sort_keys=True) if updated != decoded else value
        path, suffix = _identity_parts(value)
        if path in self.aliases:
            return self.aliases[path] + suffix
        if Path(path).is_absolute():
            mapped = self._mapped(path)
            if mapped is not None:
                return mapped + suffix
        return value

    def machine_value(self, value):
        if isinstance(value, str):
            return self._machine_string(value)
        if isinstance(value, Mapping):
            return {key: self.machine_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.machine_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.machine_value(item) for item in value)
        return value

    def metadata(self, metadata):
        result = _walk(metadata, self.resolve)
        # Numerical signatures are generated data, not arbitrary user strings.
        for key in ("raw_dgs_reduction_cache",):
            if isinstance(result.get(key), dict) and isinstance(result[key].get("signature"), str):
                result[key]["signature"] = self._machine_string(result[key]["signature"])
        return result

    def _dependencies(self, dependencies):
        if dependencies is None:
            return None
        if hasattr(dependencies, "source_ids"):
            ids = []
            namespaces = {}
            for identity in dependencies.source_ids:
                # Independent point primitives append their flattened index.
                namespace, separator, index = identity.rpartition(":")
                if separator and index.isdecimal():
                    if namespace not in namespaces:
                        namespaces[namespace] = self._machine_string(namespace)
                    updated = namespaces[namespace] + separator + index
                else:
                    updated = self._machine_string(identity)
                ids.append(updated)
            if tuple(ids) == dependencies.source_ids:
                return dependencies
            if len(set(ids)) != len(ids):
                raise ValueError("relocation merges distinct cached primitive identities; recompute from sources")
            result = copy.copy(dependencies)
            object.__setattr__(result, "source_ids", tuple(ids))
            return result
        result = copy.copy(dependencies)
        for name in ("numerator_dependencies", "exposure_dependencies"):
            object.__setattr__(result, name, self._dependencies(getattr(dependencies, name)))
        return result

    def prepare_data(self, data):
        """Rebase provenance on decode while sharing every numerical array."""
        if not hasattr(data, "with_updates") or not hasattr(data, "metadata"):
            return data
        with self._lock:
            updates = {"metadata": self.metadata(data.metadata)}
            if hasattr(data, "source_dependencies"):
                updates["source_dependencies"] = self._dependencies(data.source_dependencies)
            if hasattr(data, "counting_dependencies"):
                updates["counting_dependencies"] = self._dependencies(data.counting_dependencies)
            return data.with_updates(**updates)


def portable_project_manifest(payload, project_path, *, resolution=None):
    """Return a manifest with declared paths relative to its save destination."""
    directory = _absolute(Path(project_path).expanduser().absolute().parent, Path.cwd())
    originals, aliases = {}, {}

    def relative_path(value):
        try:
            return os.path.relpath(value, directory)
        except ValueError:
            # Files on another drive remain absolute and can use path_mappings.
            return str(value)

    def encode(value, kind):
        path, suffix = _identity_parts(value) if kind == "identity" else (value, "")
        if kind == "identity" and not Path(path).is_absolute():
            return value
        absolute = str(_absolute(path, Path.cwd()))
        relative = relative_path(absolute)
        originals[relative] = absolute
        if path != absolute:
            aliases[path] = relative
        return relative + suffix

    result = _walk(payload, encode)
    roots = {}
    if resolution is not None:
        for old, new in resolution.aliases.items():
            if Path(old).is_absolute() and old != new:
                aliases[old] = relative_path(new)
        for old, new in resolution.roots:
            roots[old] = relative_path(new)
    result[PORTABLE_PATHS_KEY] = {
        "version": 1, "origin": str(directory), "originals": originals,
        "aliases": aliases, "root_mappings": roots,
    }
    return result


def resolve_project_manifest(payload, project_path, *, path_mappings=None):
    """Resolve paths before project construction or registration of lazy caches."""
    resolution = ProjectPathResolution(project_path, recorded=payload.get(PORTABLE_PATHS_KEY),
                                       path_mappings=path_mappings)
    resolution.prime(payload)
    result = _walk(payload, resolution.resolve)
    for entry in result.get("settings", {}).get("binning_cache_entries", ()):
        if isinstance(entry, dict) and isinstance(entry.get("signature"), str):
            entry["signature"] = resolution._machine_string(entry["signature"])

    def signatures(value):
        if isinstance(value, dict):
            if "raw_dgs_reduction_cache" in value:
                cache = value["raw_dgs_reduction_cache"]
                if isinstance(cache, dict) and isinstance(cache.get("signature"), str):
                    cache["signature"] = resolution._machine_string(cache["signature"])
            for item in value.values():
                signatures(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                signatures(item)
    signatures(result.get("data_groups", ()))
    return result, resolution
