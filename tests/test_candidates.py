import pytest

from purchase_intention.modeling.candidates import (
    ESTIMATORS,
    build_candidate,
    build_candidates,
)


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_candidate_is_balanced_and_seeded(name: str) -> None:
    pipeline = build_candidate(name, {}, seed=7)

    classifier = pipeline.named_steps["classifier"]
    assert classifier.class_weight == "balanced"
    assert classifier.random_state == 7


def test_hyperparameters_are_applied() -> None:
    pipeline = build_candidate("random_forest", {"n_estimators": 12}, seed=0)

    assert pipeline.named_steps["classifier"].n_estimators == 12


def test_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="xgboost"):
        build_candidate("xgboost", {}, seed=0)


def test_rejects_unknown_hyperparameter() -> None:
    # O próprio scikit-learn recusa parâmetros que não existem.
    with pytest.raises(TypeError):
        build_candidate("random_forest", {"arvores": 10}, seed=0)


def test_builds_every_declared_candidate() -> None:
    candidates = build_candidates({"logistic_regression": {}, "random_forest": {}}, 0)

    assert list(candidates) == ["logistic_regression", "random_forest"]
