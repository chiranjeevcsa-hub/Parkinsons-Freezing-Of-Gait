from pathlib import Path

import pandas as pd

from src.build_dataset import build_dataset
from src.data_loader import RAW_COLUMN_NAMES


EXPECTED_COLUMNS = RAW_COLUMN_NAMES + [
    "participant_id",
    "recording_id",
    "source_file",
]


def write_recording(path: Path, rows: list[list[int]]) -> None:
    """Write a tiny whitespace-separated Daphnet-style recording."""
    path.write_text(
        "\n".join(" ".join(str(value) for value in row) for row in rows),
        encoding="utf-8",
    )


def valid_row(time_ms: int, label: int) -> list[int]:
    """Create one valid 11-column raw row."""
    return [time_ms, 70, 39, -970, 10, 20, -980, 30, 40, -990, label]


def test_build_dataset_creates_parquet_with_expected_columns_and_rows(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    output_path = tmp_path / "processed" / "integrated.parquet"
    write_recording(input_dir / "S01R01.txt", [valid_row(15, 0), valid_row(31, 1)])
    write_recording(input_dir / "S02R01.txt", [valid_row(15, 2)])

    build_dataset(input_dir=input_dir, output_path=output_path)

    saved = pd.read_parquet(output_path)
    assert output_path.exists()
    assert saved.columns.tolist() == EXPECTED_COLUMNS
    assert len(saved) == 3


def test_build_dataset_preserves_labels_and_metadata(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    output_path = tmp_path / "daphnet.parquet"
    write_recording(input_dir / "S03R02.txt", [valid_row(15, 0), valid_row(31, 2)])

    build_dataset(input_dir=input_dir, output_path=output_path)

    saved = pd.read_parquet(output_path)
    assert sorted(saved["annotation"].unique().tolist()) == [0, 2]
    assert saved["participant_id"].unique().tolist() == ["S03"]
    assert saved["recording_id"].unique().tolist() == ["R02"]
    assert saved["source_file"].unique().tolist() == [str(input_dir / "S03R02.txt")]


def test_build_dataset_supports_custom_input_and_output_paths(tmp_path: Path) -> None:
    custom_input = tmp_path / "custom-input"
    custom_output = tmp_path / "custom-output" / "custom-name.parquet"
    custom_input.mkdir()
    write_recording(custom_input / "S10R01.txt", [valid_row(15, 1)])

    build_dataset(input_dir=custom_input, output_path=custom_output)

    saved = pd.read_parquet(custom_output)
    assert custom_output.exists()
    assert saved.shape == (1, len(EXPECTED_COLUMNS))
    assert saved.loc[0, "participant_id"] == "S10"
