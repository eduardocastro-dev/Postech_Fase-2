"""Fixtures compartilhadas: um dataset sintético pequeno com o schema da UCI."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from purchase_intention.layers.schema import RAW_COLUMNS

N_ROWS = 200
POSITIVE_RATE = 0.2


def _behavior_columns(rng: np.random.Generator, n_rows: int) -> dict[str, np.ndarray]:
    # Valores aleatórios dentro das faixas válidas observadas na EDA.
    return {
        "Administrative": rng.integers(0, 10, n_rows),
        "Administrative_Duration": rng.uniform(0, 500, n_rows).round(2),
        "Informational": rng.integers(0, 5, n_rows),
        "Informational_Duration": rng.uniform(0, 200, n_rows).round(2),
        "ProductRelated": rng.integers(1, 50, n_rows),
        "ProductRelated_Duration": rng.uniform(0, 3000, n_rows).round(2),
        "BounceRates": rng.uniform(0, 0.2, n_rows).round(4),
        "ExitRates": rng.uniform(0, 0.2, n_rows).round(4),
        "PageValues": rng.uniform(0, 50, n_rows).round(2),
        "SpecialDay": rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], n_rows),
    }


def _context_columns(rng: np.random.Generator, n_rows: int) -> dict[str, np.ndarray]:
    return {
        "Month": rng.choice(["Feb", "Mar", "May", "June", "Nov", "Dec"], n_rows),
        "OperatingSystems": rng.integers(1, 9, n_rows),
        "Browser": rng.integers(1, 14, n_rows),
        "Region": rng.integers(1, 10, n_rows),
        "TrafficType": rng.integers(1, 21, n_rows),
        "VisitorType": rng.choice(
            ["Returning_Visitor", "New_Visitor", "Other"], n_rows
        ),
        "Weekend": rng.choice([True, False], n_rows),
    }


def make_raw_sessions(n_rows: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """Gera sessões sintéticas válidas, com um "June" e uma duplicata conhecidos."""
    rng = np.random.default_rng(seed)
    n_positive = int(n_rows * POSITIVE_RATE)
    data = _behavior_columns(rng, n_rows) | _context_columns(rng, n_rows)
    data["Revenue"] = np.arange(n_rows) < n_positive
    sessions = pd.DataFrame(data).loc[:, list(RAW_COLUMNS)]
    sessions.loc[0, "Month"] = "June"
    # A última linha copia a penúltima (ambas sem compra): exatamente 1 duplicata.
    sessions.iloc[-1] = sessions.iloc[-2]
    return sessions


@pytest.fixture
def raw_sessions() -> pd.DataFrame:
    return make_raw_sessions()


@pytest.fixture
def raw_csv_path(tmp_path: Path, raw_sessions: pd.DataFrame) -> Path:
    path = tmp_path / "raw" / "sessions.csv"
    path.parent.mkdir(parents=True)
    raw_sessions.to_csv(path, index=False)
    return path
