"""Validated loading utilities for the raw Daphnet FoG recordings."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


RAW_COLUMN_NAMES: list[str] = [
    "time_ms",
    "ankle_forward_mg",
    "ankle_vertical_mg",
    "ankle_lateral_mg",
    "thigh_forward_mg",
    "thigh_vertical_mg",
    "thigh_lateral_mg",
    "trunk_forward_mg",
    "trunk_vertical_mg",
    "trunk_lateral_mg",
    "annotation",
]

DOCUMENTED_LABELS: set[int] = {0, 1, 2}
RECORDING_FILE_PATTERN = re.compile(r"^(S\d{2})(R\d{2})\.txt$")


def load_recording(file_path: str | Path) -> pd.DataFrame:
    """Load and validate one raw Daphnet recording file.

    Args:
        file_path: Path to a recording named like ``S01R01.txt``.

    Returns:
        A DataFrame with verified raw columns plus participant, recording,
        and source-file metadata columns.

    Raises:
        FileNotFoundError: If the recording file does not exist.
        ValueError: If the file name, column count, or labels are invalid.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Recording file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Recording path is not a file: {path}")

    match = RECORDING_FILE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(
            "Recording file name must match the Daphnet pattern "
            f"'S<participant>R<recording>.txt', got: {path.name}"
        )

    participant_id, recording_id = match.groups()

    try:
        data = pd.read_csv(path, sep=r"\s+", header=None)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Recording file is empty: {path}") from error

    raw_column_count = len(data.columns)
    expected_column_count = len(RAW_COLUMN_NAMES)
    if raw_column_count != expected_column_count:
        raise ValueError(
            f"Expected {expected_column_count} raw columns in {path.name}, "
            f"found {raw_column_count}."
        )

    data.columns = RAW_COLUMN_NAMES

    invalid_labels = sorted(
        label for label in data["annotation"].dropna().unique() if label not in DOCUMENTED_LABELS
    )
    if data["annotation"].isna().any() or invalid_labels:
        raise ValueError(
            f"Unexpected annotation labels in {path.name}. "
            f"Expected only {sorted(DOCUMENTED_LABELS)}, found {invalid_labels}."
        )

    data["participant_id"] = participant_id
    data["recording_id"] = recording_id
    data["source_file"] = path.name

    return data


def load_all_recordings(data_directory: str | Path) -> pd.DataFrame:
    """Load all valid Daphnet recording files from a directory tree.

    Files are loaded in deterministic sorted order. A short summary is printed
    and also stored in ``DataFrame.attrs`` under ``recording_count`` and
    ``participant_count``.

    Args:
        data_directory: Directory containing Daphnet recording ``.txt`` files,
            either directly or in nested folders such as ``dataset/``.

    Returns:
        A single DataFrame containing all loaded recordings.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If the path is not a directory or no valid recordings exist.
    """
    directory = Path(data_directory)

    if not directory.exists():
        raise FileNotFoundError(f"Data directory does not exist: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Data path is not a directory: {directory}")

    recording_files = sorted(
        path
        for path in directory.rglob("*.txt")
        if RECORDING_FILE_PATTERN.fullmatch(path.name)
    )

    if not recording_files:
        raise ValueError(f"No valid Daphnet recording files found in: {directory}")

    recordings = [load_recording(path) for path in recording_files]
    combined = pd.concat(recordings, ignore_index=True)

    recording_count = len(recording_files)
    participant_count = combined["participant_id"].nunique()
    combined.attrs["recording_count"] = recording_count
    combined.attrs["participant_count"] = participant_count

    print(
        "Loaded "
        f"{recording_count} recordings from {participant_count} participants."
    )

    return combined
