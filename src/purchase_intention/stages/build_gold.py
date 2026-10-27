"""Estágio 3: split estratificado de treino e teste na camada Gold.

Uso: ``python -m purchase_intention.stages.build_gold``
"""

from purchase_intention.layers.gold import GoldBuilder
from purchase_intention.stages.common import bootstrap


def main() -> None:
    """Executa a construção da camada Gold."""
    params = bootstrap()
    GoldBuilder(
        input_path=params.data.silver,
        train_path=params.data.gold_train,
        test_path=params.data.gold_test,
        test_size=params.split.test_size,
        seed=params.seed,
    ).run()


if __name__ == "__main__":
    main()
