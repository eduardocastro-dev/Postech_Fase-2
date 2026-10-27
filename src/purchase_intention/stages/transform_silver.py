"""Estágio 2: validação e padronização na camada Silver.

Uso: ``python -m purchase_intention.stages.transform_silver``
"""

from purchase_intention.layers.silver import SilverTransformer
from purchase_intention.stages.common import bootstrap


def main() -> None:
    """Executa a transformação da camada Silver."""
    params = bootstrap()
    SilverTransformer(
        input_path=params.data.bronze,
        output_path=params.data.silver,
        report_path=params.data.silver_report,
    ).run()


if __name__ == "__main__":
    main()
