import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import preparar_dados
from src.metrics import calcular_metricas

from src.config import (
    LOGISTIC_MODEL,
    RANDOM_FOREST_MODEL
)

## Função regressão logistica
def treinar_logistic_regression():

    X_train, X_test, y_train, y_test = preparar_dados()
    
    resultado_log = treinar_logistic_regression(
    X_train,
    X_test,
    y_train,
    y_test
)
    


    modelo = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    y_prob = modelo.predict_proba(X_test)[:, 1]

    metricas = calcular_metricas(
        y_test,
        y_pred,
        y_prob
    )

    joblib.dump(
        modelo,
        LOGISTIC_MODEL
    )

    return {
        "modelo": modelo,
        "metricas": metricas,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_test": y_test
    }
    
## Função random Forest
def treinar_random_forest():

    X_train, X_test, y_train, y_test = preparar_dados()
    resultado_rf = treinar_random_forest(
    X_train,
    X_test,
    y_train,
    y_test
)

    modelo = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(
        X_train,
        y_train
    )

    y_pred = modelo.predict(X_test)

    y_prob = modelo.predict_proba(X_test)[:, 1]

    metricas = calcular_metricas(
        y_test,
        y_pred,
        y_prob
    )

    joblib.dump(
        modelo,
        RANDOM_FOREST_MODEL
    )

    return {
        "modelo": modelo,
        "metricas": metricas,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_test": y_test
    }