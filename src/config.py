from pathlib import Path

# ==========================
# Diretórios do Projeto
# ==========================

# Pasta raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

# Pasta de dados
DATA_DIR = BASE_DIR / "data"

# Dados brutos
RAW_DATA_DIR = DATA_DIR / "raw"

# Dados processados
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Pasta de imagens
IMAGES_DIR = BASE_DIR / "images"

# Pasta de modelos treinados
MODELS_DIR = BASE_DIR / "models"

# Dataset principal
CREDITCARD_DATA = RAW_DATA_DIR / "creditcard.csv"

# Arquivos processados

X_TRAIN_FILE = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_FILE = PROCESSED_DATA_DIR / "X_test.csv"

Y_TRAIN_FILE = PROCESSED_DATA_DIR / "y_train.csv"
Y_TEST_FILE = PROCESSED_DATA_DIR / "y_test.csv"

SCALER_FILE = MODELS_DIR / "scaler.pkl"

# Modelos treinados

LOGISTIC_MODEL = MODELS_DIR / "logistic_regression.pkl"

RANDOM_FOREST_MODEL = MODELS_DIR / "random_forest.pkl"

CONFUSION_MATRIX_IMAGE = IMAGES_DIR / "confusion_matrix.png"

ROC_CURVE_IMAGE = IMAGES_DIR / "roc_curve.png"

FEATURE_IMPORTANCE_IMAGE = IMAGES_DIR / "feature_importance.png"

MODEL_COMPARISON_IMAGE = IMAGES_DIR / "model_comparison.png"


from src.report import gerar_relatorio

print("OK")