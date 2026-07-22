from pathlib import Path
import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

RELATORIO = REPORTS_DIR / "relatorio_final.md"

def escolher_melhor_modelo(comparacao):

    melhor = comparacao.sort_values(
        by="ROC-AUC",
        ascending=False
    ).iloc[0]

    return melhor

def gerar_relatorio(comparacao):

    melhor = escolher_melhor_modelo(comparacao)

    texto = f"""
# Relatório Final

## Objetivo

Identificar fraudes em transações bancárias utilizando algoritmos de Machine Learning.

---

## Modelos Avaliados

- Logistic Regression
- Random Forest

---

## Resultado

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|---------:|----------:|-------:|---:|--------:|
"""

    for _, linha in comparacao.iterrows():

        texto += (
            f"| {linha['Modelo']} | "
            f"{linha['Accuracy']:.4f} | "
            f"{linha['Precision']:.4f} | "
            f"{linha['Recall']:.4f} | "
            f"{linha['F1']:.4f} | "
            f"{linha['ROC-AUC']:.4f} |\n"
        )

    texto += f"""

---

## Melhor Modelo

**{melhor['Modelo']}**

ROC-AUC: **{melhor['ROC-AUC']:.4f}**

---

## Conclusões

O modelo apresentou maior capacidade de distinguir transações legítimas de fraudulentas.

Em problemas de fraude bancária, métricas como Recall e ROC-AUC são mais relevantes do que Accuracy, pois o conjunto de dados é altamente desbalanceado.

"""

    with open(
        RELATORIO,
        "w",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(texto)

    print("Relatório salvo em:", RELATORIO)