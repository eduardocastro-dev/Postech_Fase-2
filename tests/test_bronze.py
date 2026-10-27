import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from purchase_intention.layers.bronze import BronzeIngestor
from purchase_intention.layers.schema import RAW_COLUMNS, DataValidationError

SOURCE_URL = "https://example.org/dataset.zip"


def _make_ingestor(source: Path, output_dir: Path) -> BronzeIngestor:
    return BronzeIngestor(
        source_path=source,
        output_path=output_dir / "sessions.parquet",
        metadata_path=output_dir / "metadata.json",
        source_url=SOURCE_URL,
    )


def test_parquet_preserves_csv_content(raw_csv_path: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "bronze"

    _make_ingestor(raw_csv_path, output_dir).run()

    bronze = pd.read_parquet(output_dir / "sessions.parquet")
    pd.testing.assert_frame_equal(bronze, pd.read_csv(raw_csv_path))


def test_metadata_records_origin_and_file_hash(
    raw_csv_path: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "bronze"

    _make_ingestor(raw_csv_path, output_dir).run()

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(raw_csv_path.read_bytes()).hexdigest()
    assert metadata["source_sha256"] == expected_hash
    assert metadata["source_url"] == SOURCE_URL
    assert metadata["rows"] == 200
    assert metadata["columns"] == list(RAW_COLUMNS)
    assert "ingested_at" in metadata


def test_repeated_runs_produce_identical_parquet(
    raw_csv_path: Path, tmp_path: Path
) -> None:
    # Idempotência: o mesmo CSV precisa gerar exatamente os mesmos bytes,
    # senão o DVC enxergaria mudança onde não houve.
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"

    _make_ingestor(raw_csv_path, first_dir).run()
    _make_ingestor(raw_csv_path, second_dir).run()

    first = (first_dir / "sessions.parquet").read_bytes()
    second = (second_dir / "sessions.parquet").read_bytes()
    assert first == second


def test_rejects_csv_with_missing_column(
    raw_sessions: pd.DataFrame, tmp_path: Path
) -> None:
    source = tmp_path / "incomplete.csv"
    raw_sessions.drop(columns="Month").to_csv(source, index=False)
    output_dir = tmp_path / "bronze"

    with pytest.raises(DataValidationError, match="Month"):
        _make_ingestor(source, output_dir).run()
    # Nenhum arquivo é gravado quando a validação falha.
    assert not output_dir.exists()
