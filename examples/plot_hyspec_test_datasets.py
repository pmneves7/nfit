from __future__ import annotations

import argparse
from pathlib import Path

from metallix import DataGroup, DatasetEntry, load_mantid_mdhisto_nxs, plot_mdhisto_auto, slice_viewer


DATASETS = {
    "1D": "1D_test.nxs",
    "2D": "2D_test.nxs",
    "4D": "4D_test.nxs",
}


def data_path(name: str) -> Path:
    path = Path("data/hyspec") / DATASETS[name]
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    return path


def load_group(names: list[str]) -> DataGroup:
    return DataGroup(
        name="hyspec_test_datasets",
        datasets=[
            DatasetEntry(
                name=name,
                data=load_mantid_mdhisto_nxs(data_path(name), copy_metadata=False),
                kind="mdhisto",
                metadata={"source": str(data_path(name))},
            )
            for name in names
        ],
    )


def non_singleton_dims(shape: tuple[int, ...]) -> list[int]:
    return [dim for dim, size in enumerate(shape) if size > 1]


def static_plot(name: str, *, channel: str):
    data = load_group([name]).get_dataset(name).data
    dims = non_singleton_dims(data.shape)
    if len(dims) == 1:
        ax = plot_mdhisto_auto(data, channel=channel)
        fig = ax.figure
    else:
        fig = plot_mdhisto_auto(
            data,
            channel=channel,
            cmap="viridis",
            color_scale="linear",
            auto_limits="min/max",
        )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot local HYSPEC MDHisto test datasets.")
    parser.add_argument(
        "datasets",
        nargs="*",
        default=None,
        metavar="{1D,2D,4D}",
        help="Datasets to plot. Defaults to all available test datasets.",
    )
    parser.add_argument("--channel", default="signal", help="signal, errors, num_events/multiplicity, or mask")
    parser.add_argument("--interactive", action="store_true", help="Open the Qt dataset-switching slice viewer")
    parser.add_argument("--save-dir", type=Path, help="Optional directory for static PNG output")
    args = parser.parse_args()
    selected_datasets = args.datasets or sorted(DATASETS)
    invalid_datasets = [name for name in selected_datasets if name not in DATASETS]
    if invalid_datasets:
        parser.error(
            "argument datasets: invalid choice: "
            f"{invalid_datasets[0]!r} (choose from {', '.join(repr(name) for name in sorted(DATASETS))})"
        )

    if not args.save_dir:
        group = load_group(selected_datasets)
        viewer = slice_viewer(
            group.data_sequence(),
            dataset_names=group.dataset_names,
            channel=args.channel,
            cmap="viridis",
            color_scale="linear",
            auto_limits="min/max",
        )
        print(f"Opened slice viewer for datasets: {', '.join(selected_datasets)}")
        viewer.run()
        return

    for name in selected_datasets:
        fig = static_plot(name, channel=args.channel)
        args.save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_dir / f"{name}_test_{args.channel}.png", dpi=150)


if __name__ == "__main__":
    main()
