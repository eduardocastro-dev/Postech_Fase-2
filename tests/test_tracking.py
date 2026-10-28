import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
import skops.io
from mlflow import MlflowClient

from purchase_intention.layers.storage import write_json
from purchase_intention.modeling.artifacts import save_model
from purchase_intention.modeling.candidates import build_candidate
from purchase_intention.modeling.evaluation import evaluate_model
from purchase_intention.modeling.preprocessing import split_features_target
from purchase_intention.modeling.training import build_cv_report, train_best_model
from purchase_intention.params import TrainParams
from purchase_intention.tracking import publisher as publisher_module
from purchase_intention.tracking.publisher import ModelPublisher, TrainingOutputs
from purchase_intention.tracking.registry import CHAMPION_ALIAS
from purchase_intention.tracking.runs import (
    flatten_cv_metrics,
    flatten_test_metrics,
    trusted_types,
)

EXPERIMENT = "test-experiment"
MODEL_NAME = "test-model"


@pytest.fixture
def tracking_uri(tmp_path: Path) -> Iterator[str]:
    # Cada teste usa um MLflow novo e isolado, apagado ao final.
    previous = mlflow.get_tracking_uri()
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri(previous)


@pytest.fixture
def publisher(tmp_path: Path, tracking_uri: str) -> ModelPublisher:
    return ModelPublisher(
        EXPERIMENT, MODEL_NAME, artifact_location=(tmp_path / "artifacts").as_uri()
    )


@pytest.fixture
def outputs(
    gold_sessions: pd.DataFrame, small_train_params: TrainParams, tmp_path: Path
) -> TrainingOutputs:
    # Reproduz em miniatura o que os estágios train e evaluate gravam.
    result = train_best_model(gold_sessions, small_train_params, seed=0)
    features, target = split_features_target(gold_sessions)
    paths = {name: tmp_path / name for name in ("model.joblib", "cv.json", "m.json")}
    save_model(result.model, paths["model.joblib"])
    write_json(build_cv_report(result, cv_folds=3), paths["cv.json"])
    write_json(evaluate_model(result.model, features, target), paths["m.json"])
    return TrainingOutputs(
        model_path=paths["model.joblib"],
        cv_results_path=paths["cv.json"],
        metrics_path=paths["m.json"],
        input_example=features.head(5),
        run_params={"seed": 0, "cv_folds": 3},
        hyperparameters=small_train_params.models,
    )


def _variant(
    outputs: TrainingOutputs, tmp_path: Path, cv_f1: float, compress: int
) -> TrainingOutputs:
    # Outro arquivo de modelo com o F1 de validação escolhido. Cada nível de
    # compressão gera bytes (e, portanto, um SHA-256) diferentes.
    model_path = tmp_path / f"model-{compress}.joblib"
    joblib.dump(joblib.load(outputs.model_path), model_path, compress=compress)
    report = json.loads(outputs.cv_results_path.read_text(encoding="utf-8"))
    report["candidates"][report["selected_model"]]["mean"]["f1"] = cv_f1
    cv_path = tmp_path / f"cv-{cv_f1}.json"
    write_json(report, cv_path)
    return replace(outputs, model_path=model_path, cv_results_path=cv_path)


def _champion_version() -> str:
    champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    return str(champion.version)


def test_flatten_metrics_use_prefixes() -> None:
    cv = flatten_cv_metrics({"mean": {"f1": 0.6}, "std": {"f1": 0.1}})
    test = flatten_test_metrics({"f1": 0.5, "confusion_matrix": {"tp": 3}})

    assert cv == {"cv_f1": 0.6, "cv_f1_std": 0.1}
    assert test == {"test_f1": 0.5, "test_tp": 3}


def test_first_publication_creates_runs_and_champion(
    publisher: ModelPublisher, outputs: TrainingOutputs
) -> None:
    publication = publisher.publish(outputs)

    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], output_format="list")
    assert publication is not None
    assert publication.version == "1"
    assert publication.is_champion
    assert len(runs) == 1 + len(outputs.hyperparameters)
    assert _champion_version() == "1"


def test_publishing_same_model_again_does_nothing(
    publisher: ModelPublisher, outputs: TrainingOutputs
) -> None:
    publisher.publish(outputs)

    assert publisher.publish(outputs) is None
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], output_format="list")
    assert len(runs) == 1 + len(outputs.hyperparameters)


def test_only_a_better_model_becomes_champion(
    publisher: ModelPublisher, outputs: TrainingOutputs, tmp_path: Path
) -> None:
    publisher.publish(_variant(outputs, tmp_path, cv_f1=0.5, compress=0))

    worse = publisher.publish(_variant(outputs, tmp_path, cv_f1=0.4, compress=1))
    assert worse is not None and not worse.is_champion
    assert _champion_version() == "1"

    better = publisher.publish(_variant(outputs, tmp_path, cv_f1=0.6, compress=2))
    assert better is not None and better.is_champion
    assert _champion_version() == "3"


def test_failed_publication_is_retried(
    publisher: ModelPublisher,
    outputs: TrainingOutputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_registry(*args: object) -> str:
        raise RuntimeError("registro indisponível")

    monkeypatch.setattr(publisher_module, "register_version", broken_registry)
    with pytest.raises(RuntimeError):
        publisher.publish(outputs)
    monkeypatch.undo()

    # A execução que falhou não conta como publicação: a nova tentativa grava tudo.
    publication = publisher.publish(outputs)
    assert publication is not None
    assert _champion_version() == "1"


def test_random_forest_loads_with_its_trusted_types(
    gold_sessions: pd.DataFrame,
) -> None:
    features, target = split_features_target(gold_sessions)
    forest = build_candidate("random_forest", {"n_estimators": 5}, 0).fit(
        features, target
    )

    loaded = skops.io.loads(skops.io.dumps(forest), trusted=trusted_types(forest))

    np.testing.assert_array_equal(loaded.predict(features), forest.predict(features))


def test_champion_predicts_like_the_original_model(
    publisher: ModelPublisher, outputs: TrainingOutputs, gold_sessions: pd.DataFrame
) -> None:
    publisher.publish(outputs)
    features, _ = split_features_target(gold_sessions)

    champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")

    original = joblib.load(outputs.model_path)
    np.testing.assert_array_equal(
        champion.predict(features), original.predict(features)
    )
