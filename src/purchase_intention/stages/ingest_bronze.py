"""Estágio 1: ingestão do CSV bruto na camada Bronze.

Uso: ``python -m purchase_intention.stages.ingest_bronze``
"""

from purchase_intention.layers.bronze import BronzeIngestor
from purchase_intention.stages.common import bootstrap


def main() -> None:
    """Executa a ingestão da camada Bronze."""
    params = bootstrap()
    BronzeIngestor(
        source_path=params.data.raw,
        output_path=params.data.bronze,
        metadata_path=params.data.bronze_metadata,
        source_url=params.data.source_url,
    ).run()


if __name__ == "__main__":
    main()
