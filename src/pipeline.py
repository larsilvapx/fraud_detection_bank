from src.preprocessing import preparar_dados

from src.models import (
    treinar_logistic_regression,
    treinar_random_forest
)


def executar_pipeline():

    print("=" * 60)
    print("INICIANDO PIPELINE")
    print("=" * 60)

    print("\nPreparando dados...")

    X_train, X_test, y_train, y_test = preparar_dados()

    print("Dados preparados.\n")

    print("Treinando Logistic Regression...")

    resultado_log = treinar_logistic_regression(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("Logistic Regression concluído.\n")

    print("Treinando Random Forest...")

    resultado_rf = treinar_random_forest(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("Random Forest concluído.\n")

    return resultado_log, resultado_rf

resultado_log, resultado_rf = executar_pipeline()

