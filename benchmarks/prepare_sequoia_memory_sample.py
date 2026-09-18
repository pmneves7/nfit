"""Create a reproducible, read-only cropped SEQUOIA histogram sample.

The input is an nfit project ZIP and the path of its nested ``.npz`` member.
The member is copied to temporary local storage before inspection, then each
NPY payload is extracted and memory-mapped separately.  Four-dimensional
channels are cropped around their center; axis value arrays are cropped to the
corresponding bin edges and scalar metadata is copied byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def _shape(value: str) -> tuple[int, int, int, int]:
    shape = tuple(int(item) for item in value.split(","))
    if len(shape) != 4 or min(shape) < 1:
        raise argparse.ArgumentTypeError("shape must be four positive integers")
    return shape  # type: ignore[return-value]


def _npy_header(stream) -> tuple[tuple[int, ...], np.dtype]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported NPY format version {version}; expected 1.0 or 2.0")
    return shape, dtype


def _copy_member(project: Path, member: str, destination: Path) -> zipfile.ZipInfo:
    with zipfile.ZipFile(project) as archive:
        info = archive.getinfo(member)
        with archive.open(info) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    return info


def _write_array(output: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    with output.open(name, "w", force_zip64=True) as stream:
        np.lib.format.write_array(stream, array, allow_pickle=False)


def _copy_zip_entry(source: zipfile.ZipFile, output: zipfile.ZipFile, info: zipfile.ZipInfo) -> None:
    with source.open(info) as input_stream, output.open(info.filename, "w", force_zip64=True) as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)


def _load_mapped(source: zipfile.ZipFile, info: zipfile.ZipInfo, directory: Path) -> tuple[Path, np.memmap]:
    extracted = directory / info.filename.replace("/", "_")
    with source.open(info) as input_stream, extracted.open("wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
    return extracted, np.load(extracted, mmap_mode="r", allow_pickle=False)


def prepare(project: Path, member: str, output: Path, target_shape: tuple[int, int, int, int], temporary_dir: Path | None, compression: int) -> dict[str, object]:
    project = project.resolve()
    output = output.resolve()
    if output == project:
        raise ValueError("output must not overwrite the project archive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stat = project.stat()
    with tempfile.TemporaryDirectory(prefix="nfit-sequoia-sample-", dir=temporary_dir or output.parent) as temporary:
        directory = Path(temporary)
        nested_path = directory / "source-member.npz"
        outer_info = _copy_member(project, member, nested_path)
        with zipfile.ZipFile(nested_path) as source:
            infos = [info for info in source.infolist() if not info.is_dir()]
            by_name = {info.filename: info for info in infos}
            signal_info = by_name.get("signal.npy")
            if signal_info is None:
                raise ValueError("nested archive does not contain signal.npy")
            with source.open(signal_info) as signal_stream:
                source_shape, _signal_dtype = _npy_header(signal_stream)
            if len(source_shape) != 4:
                raise ValueError(f"signal must be four-dimensional, got {source_shape}")
            if any(target > source for target, source in zip(target_shape, source_shape, strict=True)):
                raise ValueError(f"requested shape {target_shape} exceeds source shape {source_shape}")
            starts = tuple((source - target) // 2 for source, target in zip(source_shape, target_shape, strict=True))
            stops = tuple(start + target for start, target in zip(starts, target_shape, strict=True))
            selection = tuple(slice(start, stop) for start, stop in zip(starts, stops, strict=True))
            cropped, axis_cropped, copied = [], [], []
            with zipfile.ZipFile(output, "x", compression=compression, compresslevel=6 if compression == zipfile.ZIP_DEFLATED else None) as destination:
                for info in infos:
                    if not info.filename.endswith(".npy"):
                        _copy_zip_entry(source, destination, info)
                        copied.append(info.filename)
                        continue
                    with source.open(info) as header_stream:
                        header_shape, _dtype = _npy_header(header_stream)
                    if header_shape == source_shape:
                        extracted, array = _load_mapped(source, info, directory)
                        _write_array(destination, info.filename, np.asarray(array[selection]))
                        del array
                        extracted.unlink()
                        cropped.append(info.filename.removesuffix(".npy"))
                    elif info.filename.startswith("axis_") and info.filename.endswith("_values.npy"):
                        index_text = info.filename.removeprefix("axis_").split("_", 1)[0]
                        if index_text.isdigit() and int(index_text) < 4 and header_shape == (source_shape[int(index_text)] + 1,):
                            axis = int(index_text)
                            extracted, array = _load_mapped(source, info, directory)
                            _write_array(destination, info.filename, np.asarray(array[starts[axis]:stops[axis] + 1]))
                            del array
                            extracted.unlink()
                            axis_cropped.append(info.filename.removesuffix(".npy"))
                            continue
                        _copy_zip_entry(source, destination, info)
                        copied.append(info.filename)
                    else:
                        _copy_zip_entry(source, destination, info)
                        copied.append(info.filename)
    with zipfile.ZipFile(output) as validation:
        for name in cropped:
            with validation.open(f"{name}.npy") as header_stream:
                shape, _dtype = _npy_header(header_stream)
            if shape != target_shape:
                raise ValueError(f"output {name} shape {shape} does not match {target_shape}")
    return {
        "source": {"project": str(project), "project_bytes": stat.st_size, "project_mtime_ns": stat.st_mtime_ns,
                   "member": member, "member_compressed_bytes": outer_info.compress_size, "member_uncompressed_bytes": outer_info.file_size},
        "output": {"path": str(output), "bytes": output.stat().st_size},
        "source_shape": list(source_shape), "target_shape": list(target_shape),
        "slice_starts": list(starts), "slice_stops": list(stops),
        "cropped_channels": cropped, "cropped_axis_values": axis_cropped,
        "metadata_entries_copied": copied,
        "method": "outer ZIP member copied to temporary NPZ; each NPY payload extracted and memory-mapped sequentially",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", type=_shape, default=(96, 96, 96, 48))
    parser.add_argument("--temporary-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--compression", choices=("stored", "deflated"), default="stored")
    args = parser.parse_args()
    compression = zipfile.ZIP_STORED if args.compression == "stored" else zipfile.ZIP_DEFLATED
    target = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    if target.resolve() == args.project.resolve():
        parser.error("manifest must not overwrite the project archive")
    manifest = prepare(args.project, args.member, args.output, args.shape, args.temporary_dir, compression)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
