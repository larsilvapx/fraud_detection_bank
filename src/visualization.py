import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from src.config import (
    CONFUSION_MATRIX_IMAGE,
    ROC_CURVE_IMAGE,
    FEATURE_IMPORTANCE_IMAGE,
    MODEL_COMPARISON_IMAGE
)

def plot_confusion_matrix(modelo, X_test, y_test):

    fig, ax = plt.subplots(figsize=(6,6))

    ConfusionMatrixDisplay.from_estimator(
        modelo,
        X_test,
        y_test,
        cmap="Blues",
        ax=ax
    )

    plt.title("Matriz de Confusão")

    plt.savefig(
        CONFUSION_MATRIX_IMAGE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    
    def plot_roc_curve(modelo, X_test, y_test):

    fig, ax = plt.subplots(figsize=(8,6))

    RocCurveDisplay.from_estimator(
        modelo,
        X_test,
        y_test,
        ax=ax
    )

    plt.title("Curva ROC")

    plt.savefig(
        ROC_CURVE_IMAGE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    
def plot_metricas(df):

    plt.figure(figsize=(12,6))

    df.set_index("Modelo").plot(kind="bar")

    plt.title("Comparação entre Modelos")

    plt.ylabel("Score")

    plt.xticks(rotation=0)

    plt.savefig(
        MODEL_COMPARISON_IMAGE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    
def plot_feature_importance(modelo, X_train):

    import pandas as pd

    importancia = pd.Series(
        modelo.feature_importances_,
        index=X_train.columns
    )

    importancia = importancia.sort_values(ascending=False)

    plt.figure(figsize=(10,8))

    importancia.head(15).plot(kind="barh")

    plt.title("15 Variáveis Mais Importantes")

    plt.gca().invert_yaxis()

    plt.savefig(
        FEATURE_IMPORTANCE_IMAGE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()    
    
