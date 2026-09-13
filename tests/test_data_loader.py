from pathlib import Path

import pytest

from src.data_loader import RAW_COLUMN_NAMES, load_all_recordings, load_recording


def write_recording(path: Path, rows: list[list[int]]) -> None:
    """Write a small whitespace-separated recording fixture."""
    path.write_text(
        "\n".join(" ".join(str(value) for value in row) for row in rows),
        encoding="utf-8",
    )


def valid_row(label: int = 1) -> list[int]:
    """Create one valid 11-column Daphnet-style row."""
    return [15, 70, 39, -970, 10, 20, -980, 30, 40, -990, label]


def test_load_recording_assigns_expected_columns(tmp_path: Path) -> None:
    recording_path = tmp_path / "S01R01.txt"
    write_recording(recording_path, [valid_row(0), valid_row(1), valid_row(2)])

    data = load_recording(recording_path)

    assert data.columns[: len(RAW_COLUMN_NAMES)].tolist() == RAW_COLUMN_NAMES
    assert data.loc[0, "time_ms"] == 15
    assert data.loc[0, "ankle_forward_mg"] == 70
    assert data.loc[0, "annotation"] == 0


def test_load_recording_extracts_participant_and_recording_ids(tmp_path: Path) -> None:
    recording_path = tmp_path / "S03R02.txt"
    write_recording(recording_path, [valid_row()])

    data = load_recording(recording_path)

    assert data["participant_id"].unique().tolist() == ["S03"]
    assert data["recording_id"].unique().tolist() == ["R02"]
    assert data["source_file"].unique().tolist() == [
    recording_path.name
]


def test_load_recording_rejects_wrong_number_of_columns(tmp_path: Path) -> None:
    recording_path = tmp_path / "S01R01.txt"
    write_recording(recording_path, [[15, 70, 39]])

    with pytest.raises(ValueError, match="Expected 11 raw columns"):
        load_recording(recording_path)


def test_load_recording_rejects_unexpected_labels(tmp_path: Path) -> None:
    recording_path = tmp_path / "S01R01.txt"
    write_recording(recording_path, [valid_row(3)])

    with pytest.raises(ValueError, match="Unexpected annotation labels"):
        load_recording(recording_path)


def test_load_all_recordings_combines_multiple_files(tmp_path: Path) -> None:
    write_recording(tmp_path / "S02R01.txt", [valid_row(1), valid_row(2)])
    write_recording(tmp_path / "S01R01.txt", [valid_row(0)])

    combined = load_all_recordings(tmp_path)

    assert len(combined) == 3
    assert combined["participant_id"].tolist() == ["S01", "S02", "S02"]
    assert combined["recording_id"].tolist() == ["R01", "R01", "R01"]
    assert combined.attrs["recording_count"] == 2
    assert combined.attrs["participant_count"] == 2
