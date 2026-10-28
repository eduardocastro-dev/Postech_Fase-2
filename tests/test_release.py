"""Consistência dos resultados de entrega armazenados pelo DVC."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK = yaml.safe_load((PROJECT_ROOT / "dvc.lock").read_text(encoding="utf-8"))
REMOTE = PROJECT_ROOT / "dvc-storage/files/md5"


def _versioned_report(stage: str, path: str) -> dict:
    outputs = LOCK["stages"][stage]["outs"]
    output = next(item for item in outputs if item["path"] == path)
    md5 = output["md5"]
    cache_file = REMOTE / md5[:2] / md5[2:]
    content = cache_file.read_bytes()
    assert hashlib.md5(content, usedforsecurity=False).hexdigest() == md5
    return json.loads(content)


def test_versioned_results_match_documented_release() -> None:
    metrics = _versioned_report("evaluate", "reports/metrics.json")
    assert metrics["f1"] == pytest.approx(0.6616541353383458)
    assert metrics["roc_auc"] == pytest.approx(0.9235850810463165)
    assert metrics["accuracy"] == pytest.approx(0.8905109489051095)
    assert metrics["confusion_matrix"] == {"tn": 1932, "fp": 152, "fn": 118, "tp": 264}


def test_versioned_selection_match_documented_release() -> None:
    report = _versioned_report("train", "reports/cv_results.json")
    assert report["selected_model"] == "random_forest"
    assert report["cv_folds"] == 5
    assert report["threshold"] == pytest.approx(0.586776)
    assert report["candidates"]["random_forest"]["mean"]["f1"] == pytest.approx(
        0.678868
    )
