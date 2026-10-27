import importlib.util
import re
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load(
    (PROJECT_ROOT / "configs" / "params.yaml").read_text(encoding="utf-8")
)
STAGES = yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text(encoding="utf-8"))[
    "stages"
]
VARIABLE = re.compile(r"^\$\{(.+)\}$")


def _lookup(dotted_key: str) -> object:
    # Navega no params.yaml por chaves como "split.test_size".
    value: object = PARAMS
    for key in dotted_key.split("."):
        assert isinstance(value, dict) and key in value, dotted_key
        value = value[key]
    return value


def _stage_params(stage: dict) -> list[str]:
    return [
        key
        for group in stage.get("params", [])
        for keys in group.values()
        for key in keys
    ]


@pytest.mark.parametrize("name", list(STAGES))
def test_stage_command_points_to_existing_module(name: str) -> None:
    module = STAGES[name]["cmd"].removeprefix("python -m ")

    assert importlib.util.find_spec(module) is not None


@pytest.mark.parametrize("name", list(STAGES))
def test_stage_paths_exist_or_come_from_params(name: str) -> None:
    stage = STAGES[name]
    for path in [*stage["deps"], *stage["outs"]]:
        match = VARIABLE.match(path)
        if match:
            # Dados não existem num clone recém-feito: basta a chave ser válida.
            assert isinstance(_lookup(match.group(1)), str)
        else:
            assert (PROJECT_ROOT / path).is_file(), path


@pytest.mark.parametrize("name", list(STAGES))
def test_stage_params_exist_in_params_file(name: str) -> None:
    for key in _stage_params(STAGES[name]):
        _lookup(key)
