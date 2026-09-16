# src/features/build_pipeline.py

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Construye el preprocesador que aplica:
    - A numéricas: imputación por mediana + escalado estándar.
    - A categóricas: imputación por valor más frecuente + one-hot encoding.
    """
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, NUMERIC_FEATURES),
        ('cat', categorical_pipe, CATEGORICAL_FEATURES)
    ])

    return preprocessor