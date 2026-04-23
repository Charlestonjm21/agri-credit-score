"""CLI integration tests for the generator and train commands."""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from agri_credit_score.models.train import cli as train_cli
from agri_credit_score.synthetic.generator import cli as generator_cli


def test_cli_end_to_end_time_split(tmp_path: Path) -> None:
    runner = CliRunner()
    data_path = tmp_path / "borrowers.parquet"
    models_path = tmp_path / "models"

    gen_result = runner.invoke(
        generator_cli,
        ["--n", "500", "--spread-months", "24", "--out", str(data_path)],
    )
    assert gen_result.exit_code == 0, f"Generator CLI failed:\n{gen_result.output}"

    train_result = runner.invoke(
        train_cli,
        [
            "--data", str(data_path),
            "--out", str(models_path),
            "--split-mode", "time",
            "--trials", "3",
        ],
    )
    assert train_result.exit_code == 0, f"Train CLI failed:\n{train_result.output}"

    metadata_path = models_path / "metadata.json"
    assert metadata_path.exists(), "metadata.json not written"
    with open(metadata_path) as f:
        metadata = json.load(f)

    xgb = metadata["xgboost"]
    assert "time_split_metrics" in xgb, f"Missing time_split_metrics: {list(xgb)}"
    assert "random_split_metrics" in xgb, f"Missing random_split_metrics: {list(xgb)}"
    assert "future_holdout_metrics" in xgb, f"Missing future_holdout_metrics: {list(xgb)}"
