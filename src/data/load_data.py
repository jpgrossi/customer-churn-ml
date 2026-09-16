import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ID_COL, TARGET, RANDOM_STATE, TEST_SIZE


def load_data(path):
    df = pd.read_csv(path)
    return df


def split_data(df):
    X = df.drop(columns=[TARGET, ID_COL])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
