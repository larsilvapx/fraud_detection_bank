import pandas as pd
from pathlib import Path

from src.config import CREDITCARD_DATA


def carregar_dados():

    if not Path(CREDITCARD_DATA).exists():

        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{CREDITCARD_DATA}"
        )

    df = pd.read_csv(CREDITCARD_DATA)

    print("=" * 60)
    print("Dataset carregado com sucesso")
    print(f"Linhas : {df.shape[0]}")
    print(f"Colunas: {df.shape[1]}")
    print("=" * 60)

    return df