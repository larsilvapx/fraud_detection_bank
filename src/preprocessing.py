from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import carregar_dados

import pandas as pd
import joblib

from src.config import SCALER_FILE


def preparar_dados():

    df = carregar_dados()

    X = df.drop("Class", axis=1)

    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_FILE)

    X_train = pd.DataFrame(
        X_train,
        columns=X.columns
    )

    X_test = pd.DataFrame(
        X_test,
        columns=X.columns
    )

    return X_train, X_test, y_train, y_test