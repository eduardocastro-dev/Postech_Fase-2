from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from purchase_intention.params import load_params

PROJECT_PARAMS = Path(__file__).resolve().parents[1] / "configs" / "params.yaml"


def _write_params(tmp_path: Path, **overrides: object) -> Path:
    # Parte do arquivo real do projeto e altera apenas o que o teste precisa.
    content = yaml.safe_load(PROJECT_PARAMS.read_text(encoding="utf-8"))
    content.update(overrides)
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def test_project_params_file_is_valid() -> None:
    params = load_params(PROJECT_PARAMS)

    assert params.seed == 42
    assert params.split.test_size == 0.2
    assert params.data.raw == Path("data/raw/online_shoppers_intention.csv")


@pytest.mark.parametrize("test_size", [0, 1, 1.5])
def test_rejects_test_size_outside_open_interval(
    tmp_path: Path, test_size: float
) -> None:
    path = _write_params(tmp_path, split={"test_size": test_size})

    with pytest.raises(ValidationError):
        load_params(path)


def test_rejects_cross_validation_with_less_than_two_folds(tmp_path: Path) -> None:
    content = yaml.safe_load(PROJECT_PARAMS.read_text(encoding="utf-8"))
    path = _write_params(tmp_path, train={**content["train"], "cv_folds": 1})

    with pytest.raises(ValidationError, match="cv_folds"):
        load_params(path)


def test_rejects_unknown_keys(tmp_path: Path) -> None:
    # Pega erros de digitação, como "sed" em vez de "seed".
    path = _write_params(tmp_path, sed=1)

    with pytest.raises(ValidationError, match="sed"):
        load_params(path)
