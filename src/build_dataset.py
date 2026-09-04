"""Build the integrated validated Daphnet dataset as a Parquet file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import DOCUMENTED_LABELS, load_all_recordings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "dataset_fog_release" / "dataset"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "daphnet_integrated.parquet"
EXPECTED_COLUMN_COUNT = 14


def build_dataset(input_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Load validated raw recordings and save the integrated dataset.

    Args:
        input_dir: Directory containing raw Daphnet recording files.
        output_path: Destination Parquet path.

    Returns:
        The combined DataFrame that was saved.

    Raises:
        ValueError: If the combined data does not have the expected schema or
            contains undocumented labels.
    """
    data = load_all_recordings(input_dir)

    if len(data.columns) != EXPECTED_COLUMN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COLUMN_COUNT} columns after loading, "
            f"found {len(data.columns)}."
        )

    observed_labels = set(data["annotation"].unique())
    unexpected_labels = sorted(observed_labels - DOCUMENTED_LABELS)
    if unexpected_labels:
        raise ValueError(
            "Integrated dataset contains undocumented labels: "
            f"{unexpected_labels}."
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(destination, index=False)

    label_counts = data["annotation"].value_counts().sort_index()
    participant_count = data["participant_id"].nunique()
    recording_count = data["source_file"].nunique()
    saved_size = destination.stat().st_size

    print(f"Output path: {destination}")
    print(f"DataFrame shape: {data.shape}")
    print(f"Participant count: {participant_count}")
    print(f"Recording count: {recording_count}")
    print("Label counts:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    print(f"Saved file size: {saved_size} bytes")

    return data


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset materialization."""
    parser = argparse.ArgumentParser(
        description="Build the validated integrated Daphnet dataset as Parquet."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Raw Daphnet recording directory. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output Parquet path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line build."""
    args = parse_args()
    build_dataset(input_dir=args.input_dir, output_path=args.output_path)


if __name__ == "__main__":
    main()
