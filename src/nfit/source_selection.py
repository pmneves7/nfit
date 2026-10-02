"""Bounded run expressions and metadata-only source selection previews.

GRASP-style operators describe two acquisition grouping indices: ``depth`` and
``stack``. Sources in one slot are summed; separate slots remain separate
measurements. Repeated sources keep the same canonical identity and must not be
mistaken for statistically independent acquisitions by downstream importers.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_SOURCE_APPEARANCES = 100_000
MAX_SOURCE_NUMBER_PADDING = 64
RUN_EXPRESSION_HELP = """Run expressions (inclusive ranges):
  10,12,15       separate expression groups
  10:15          one group per run; 10:::19 uses stride three
  10:3:19        equivalent legacy start:stride:end spelling
  10+12          sum those runs in one group
  10>15          sum the range in one group; 10>>>19 uses stride three
  10'12          advance the second grouping index
  10;15          range through the second grouping index (;;; is stride three)
  10(n;s1;s2;r)  sum n sources per block, all blocks in one group
  10[n;s1;s2;r]  sum n sources per group, one depth group per block
  10{n;s1;s2;r}  one depth group per source
  10/n;s1;s2;r/  one second-index group per source
Block parameters: n = sources per block, s1 = files skipped within a block,
s2 = files skipped between blocks, r = repeat count. Defaults are n;0;0;1.
  10|3|          repeat the same source in three depth groups
  10!3!          repeat the same source in three second-index groups
  0 or 0|5|      reserve one or five empty groups
  x              clear the source selection
Repeated sources retain one physical source identity; repetition is not a new
independent measurement. Separate colon ranges with commas rather than chaining
them, because start:stride:end takes precedence. Expansion is limited to 100,000
source appearances. Preview missing files and grouped sources before import.
Import is flat by default. Optional expression groups become ordinary nfit
DatasetGroup subfolders; they do not create another project container type.
"""


@dataclass(frozen=True)
class SourceSelection:
    """Saved naming convention and original run expression for a selection."""

    directory: str | Path
    prefix: str
    suffix: str
    expression: str
    padding: int = 0

    def __post_init__(self):
        if isinstance(self.padding, bool) or not isinstance(self.padding, int) or not 0 <= self.padding <= MAX_SOURCE_NUMBER_PADDING:
            raise ValueError(f"number padding must be an integer from zero to {MAX_SOURCE_NUMBER_PADDING}")
        if not str(self.directory).strip():
            raise ValueError("select a data directory")
        object.__setattr__(self, "directory", str(self.directory))
        for name, value in (("prefix", self.prefix), ("suffix", self.suffix)):
            if not isinstance(value, str) or any(character in value for character in ("/", "\\", "\x00")):
                raise ValueError(f"file {name} must be a filename fragment without directory separators")
        if not isinstance(self.expression, str):
            raise TypeError("run expression must be text")

    def to_dict(self) -> dict[str, Any]:
        return {"mode": "expression", "directory": str(self.directory), "prefix": self.prefix,
                "suffix": self.suffix, "expression": self.expression, "padding": self.padding}

    @classmethod
    def from_dict(cls, payload):
        """Read a saved selector, including the earlier ``numors`` spelling."""
        return cls(payload["directory"], payload.get("prefix", ""), payload.get("suffix", ""),
                   payload.get("expression", payload.get("numors", "")), payload.get("padding", 0))


@dataclass(frozen=True)
class SourceAppearance:
    """One source occurrence (or an empty slot) with its two grouping indices."""

    run_number: int | None
    depth: int
    stack: int
    path: str | None = None
    source_identity: str | None = None

    @property
    def slot(self):
        return self.depth, self.stack

    def to_dict(self):
        return {"run_number": self.run_number, "depth": self.depth, "stack": self.stack,
                "path": self.path, "source_identity": self.source_identity}


@dataclass(frozen=True)
class SourceSlot:
    """Sources to combine in one acquisition slot, in expression order."""

    depth: int
    stack: int
    run_numbers: tuple[int | None, ...]
    appearance_indices: tuple[int, ...]

    @property
    def empty(self):
        return all(value is None for value in self.run_numbers)

    def to_dict(self):
        return {"depth": self.depth, "stack": self.stack, "run_numbers": list(self.run_numbers),
                "appearance_indices": list(self.appearance_indices), "empty": self.empty}


@dataclass(frozen=True)
class SourceSelectionPlan:
    """Expanded source grouping and filesystem inventory, without event arrays."""

    selection: SourceSelection
    appearances: tuple[SourceAppearance, ...]
    slots: tuple[SourceSlot, ...]
    resolved_files: tuple[str, ...]
    missing: tuple[str, ...]
    duplicates: tuple[str, ...]
    metadata: Mapping[str, Mapping[str, Any]]

    @property
    def ready(self):
        return bool(self.resolved_files) and not self.missing

    @property
    def clear(self):
        return self.selection.expression.strip().lower() == "x"

    def to_dict(self):
        return {**self.selection.to_dict(), "appearances": [item.to_dict() for item in self.appearances],
                "slots": [item.to_dict() for item in self.slots], "resolved_files": list(self.resolved_files),
                "missing": list(self.missing), "duplicates": list(self.duplicates),
                "metadata": {key: dict(value) for key, value in self.metadata.items()}}


class _ExpressionParser:
    def __init__(self, text, limit):
        self.text = re.sub(r"\s+", "", text)
        self.limit = limit
        self.position = 0
        self.depth = self.stack = 1
        self.last = 0
        self.items = []

    def error(self, message):
        raise ValueError(f"invalid run expression at character {self.position + 1}: {message}")

    def capacity(self, additional):
        if additional > self.limit - len(self.items):
            raise ValueError(f"run expression exceeds the {self.limit:,}-appearance expansion limit")

    def integer(self):
        match = re.match(r"\d+", self.text[self.position:])
        if match is None:
            self.error("expected a nonnegative run number")
        token = match.group()
        if len(token) > 64:
            self.error("run number is too long")
        self.position += len(token)
        return int(token)

    def append(self, number):
        self.capacity(1)
        self.items.append(SourceAppearance(None if number == 0 else number, self.depth, self.stack))
        self.last = number

    def range(self, endpoint, step, axis):
        if step == 0 or (endpoint - self.last)*step < 0:
            self.error("range stride must be nonzero and point toward its endpoint")
        self.capacity(abs(endpoint - self.last)//abs(step))
        numbers = range(self.last + step, endpoint + (1 if step > 0 else -1), step)
        for number in numbers:
            if axis == "depth":
                self.depth += 1
                self.stack = 1
            elif axis == "stack":
                self.stack += 1
            self.append(number)

    def bracket(self, opening):
        closing = {"(": ")", "[": "]", "{": "}", "/": "/", "|": "|", "!": "!"}[opening]
        self.position += 1
        end = self.text.find(closing, self.position)
        if end < 0:
            self.error(f"missing closing {closing!r}")
        text = self.text[self.position:end]
        if not re.fullmatch(r"\d+(?:;\d+){0,3}", text):
            self.error("block parameters must be n[;skip_within;skip_between;repeat]")
        if any(len(item) > 64 for item in text.split(";")):
            self.error("block parameter is too long")
        parameters = [int(item) for item in text.split(";")]
        self.position = end + 1
        count = parameters[0]
        if count <= 0:
            self.error("block/repeat count must be positive")
        if opening in {"|", "!"}:
            if len(parameters) != 1:
                self.error("source repetition accepts one count")
            self.capacity(count - 1)
            for _ in range(count - 1):
                if opening == "|":
                    self.depth += 1
                else:
                    self.stack += 1
                self.append(self.last)
            return
        parameters += [0, 0, 1][len(parameters) - 1:]
        count, skip_within, skip_between, repeat = parameters
        if repeat <= 0:
            self.error("block repeat count must be positive")
        # The preceding seed belongs to the first requested slot. GRASP's
        # MATLAB implementation forgets to remove its fourth-index entry and
        # starts /.../ one slot late; retain the intended first-slot behavior.
        seed = self.items.pop()
        self.depth, self.stack = seed.depth, seed.stack
        self.capacity(count*repeat)
        number = self.last
        for block in range(repeat):
            if block and opening == "[":
                self.depth += 1
            for within in range(count):
                if (block or within) and opening == "{":
                    self.depth += 1
                elif (block or within) and opening == "/":
                    self.stack += 1
                self.append(number)
                if within != count - 1:
                    number += 1 + skip_within
            if block != repeat - 1:
                number += 1 + skip_between

    def parse(self):
        if not self.text:
            raise ValueError("enter at least one run number or expression")
        if self.text.lower() == "x":
            return ()
        self.append(self.integer())
        while self.position < len(self.text):
            operator = self.text[self.position]
            if operator in "([{ /|!".replace(" ", ""):
                self.bracket(operator)
                continue
            if operator not in ",+';:>":
                self.error(f"unsupported operator {operator!r}")
            self.position += 1
            if operator in {",", "+", "'"}:
                if operator == ",":
                    self.depth += 1
                    self.stack = 1
                elif operator == "'":
                    self.stack += 1
                self.append(self.integer())
                continue
            skip = 1
            while self.position < len(self.text) and self.text[self.position] == operator:
                skip += 1
                self.position += 1
            axis = {":": "depth", ";": "stack", ">": "sum"}[operator]
            # Preserve nfit's earlier start:step:end spelling. Repeated colons
            # remain the GRASP stride spelling, e.g. 10:::19 has stride three.
            if operator == ":" and skip == 1:
                if re.match(r"[+-]?\d+:\d+:", self.text[self.position:]):
                    self.error("separate chained colon ranges with commas; start:stride:end has three fields")
                match = re.match(r"([+-]?\d+):(\d+)(?=$|[,+';>\[({/|!])", self.text[self.position:])
                if match is not None:
                    step = int(match.group(1))
                    endpoint = int(match.group(2))
                    self.position += len(match.group())
                    self.range(endpoint, step, axis)
                    continue
            endpoint = self.integer()
            self.range(endpoint, skip if endpoint >= self.last else -skip, axis)
        return tuple(self.items)


def parse_run_expression(expression: str, *, max_appearances=MAX_SOURCE_APPEARANCES) -> tuple[SourceAppearance, ...]:
    """Expand run expressions into explicit acquisition slots, without I/O.

    Supported forms: lists ``,``; inclusive individual ranges ``:``; summed
    ranges ``>``; repeated operators specify stride; ``+`` sums sources in one
    slot. Apostrophe and semicolon advance the second grouping index. Blocks
    ``(n;s1;s2;r)`` sum cumulatively, ``[n;s1;s2;r]`` sum per depth,
    ``{n;s1;s2;r}`` create individual depths, and ``/n;s1;s2;r/`` create stacks.
    ``|n|`` and ``!n!`` repeat a source in depth or stack (including its first
    appearance). ``0`` is an empty slot; ``x`` clears the selection.

    Existing ``start:step:end`` ranges remain supported. They take precedence
    over chaining two single-colon ranges; use commas to separate such ranges.
    Expansion is bounded before a large range/block allocates its entries.
    """
    if not isinstance(expression, str):
        raise TypeError("run expression must be text")
    if isinstance(max_appearances, bool) or not isinstance(max_appearances, int) or not 1 <= max_appearances <= MAX_SOURCE_APPEARANCES:
        raise ValueError(f"maximum appearances must be from 1 to {MAX_SOURCE_APPEARANCES:,}")
    return _ExpressionParser(expression, max_appearances).parse()


def _source_slots(appearances):
    grouped = {}
    for index, appearance in enumerate(appearances):
        grouped.setdefault(appearance.slot, []).append(index)
    return tuple(SourceSlot(depth, stack, tuple(appearances[index].run_number for index in indices), tuple(indices))
                 for (depth, stack), indices in grouped.items())


def _preview_metadata(path):
    """Read bounded acquisition headers, never event or histogram arrays."""
    source = Path(path)
    stat = source.stat()
    record = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if source.suffix.lower() not in {".nxs", ".h5", ".hdf5"} and not source.name.endswith(".nxs.h5"):
        return record
    try:
        import h5py

        with h5py.File(source, "r") as handle:
            entry = handle.get("entry")
            if entry is not None:
                def scalar(paths):
                    for candidate in paths:
                        dataset = entry.get(candidate)
                        if not isinstance(dataset, h5py.Dataset) or dataset.size != 1 or dataset.dtype.itemsize > 65536:
                            continue
                        value = dataset[()]
                        if hasattr(value, "item"):
                            value = value.item()
                        if isinstance(value, bytes):
                            value = value.decode("utf-8", errors="replace")
                        if not isinstance(value, (str, int, float, bool)):
                            value = str(value)
                        if isinstance(value, float) and not math.isfinite(value):
                            continue
                        return value
                    return None

                for key in ("run_number", "title", "start_time", "end_time"):
                    value = scalar((key,))
                    if value is not None:
                        record[key] = value
                headers = {
                    "instrument_name": ("instrument/name", "instrument_name"),
                    "incident_energy_meV": ("Ei", "DASlogs/Ei/average_value", "DASlogs/EnergyRequest/average_value", "DASlogs/EnergyRequest/value"),
                    "t0_microseconds": ("T0", "DASlogs/T0/average_value", "DASlogs/T0/value"),
                    "temperature_K": ("temperature", "DASlogs/SampleTemp/average_value", "DASlogs/temperature/average_value", "DASlogs/temperature/value"),
                }
                for key, candidates in headers.items():
                    value = scalar(candidates)
                    if value is not None:
                        record[key] = value
                record["event_count"] = sum(int(group["event_id"].shape[0])
                    for name, group in entry.items() if name.endswith("_events") and "event_id" in group)
    except (ImportError, OSError):
        record["header_status"] = "not a readable HDF5 source"
    return record


def resolve_source_selection(selection, *, inspect_metadata=False, metadata_reader=None,
                             max_appearances=MAX_SOURCE_APPEARANCES) -> SourceSelectionPlan:
    """Resolve names, grouping, missing files and intentional duplicate identities.

    The default checks file existence only. Optional lightweight metadata is
    read once per unique source; a custom ``metadata_reader(path)`` can expose an
    instrument's acquisition headers. No reducer or event array is invoked.
    Missing sources are reported rather than silently omitted from the preview.
    """
    if isinstance(selection, Mapping):
        selection = SourceSelection.from_dict(selection)
    if not isinstance(selection, SourceSelection):
        raise TypeError("selection must be a SourceSelection or saved selector mapping")
    parsed = parse_run_expression(selection.expression, max_appearances=max_appearances)
    directory = Path(selection.directory).expanduser().resolve()
    appearances = []
    for item in parsed:
        if item.run_number is None:
            appearances.append(item)
            continue
        number = str(item.run_number).zfill(selection.padding)
        path = str((directory / f"{selection.prefix}{number}{selection.suffix}").resolve())
        appearances.append(SourceAppearance(item.run_number, item.depth, item.stack, path, path))
    appearances = tuple(appearances)
    identities = Counter(item.source_identity for item in appearances if item.source_identity is not None)
    resolved = []
    missing = []
    metadata = {}
    for path in identities:
        if not Path(path).is_file():
            missing.append(path)
        else:
            resolved.append(path)
            if inspect_metadata:
                try:
                    metadata[path] = dict((metadata_reader or _preview_metadata)(path))
                except (OSError, ValueError) as error:
                    metadata[path] = {"header_error": str(error)}
    return SourceSelectionPlan(selection, appearances, _source_slots(appearances), tuple(resolved), tuple(missing),
                               tuple(path for path, count in identities.items() if count > 1), metadata)
