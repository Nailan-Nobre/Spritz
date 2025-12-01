import numpy as np
import pandas as pd


def fetch_data(source: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Fetch data from the given source.

    Args:
        source (str): The data source URL or file path.

    Returns:
        tuple[np.ndarray, np.ndarray, list[str]]: Features (X), target (y) arrays, and feature names.
    """
    df = pd.read_csv(source)

    # Selecionar apenas colunas numéricas
    numeric_df = df.select_dtypes(include=[np.number])

    # Obter nomes das colunas
    feature_names = numeric_df.columns.tolist()

    # Converter para numpy array
    data = numeric_df.to_numpy()

    # Separar features (X) e target (y)
    X = data[:, :-1]
    y = data[:, -1]

    # Nomes das features (excluindo a última coluna que é o target)
    X_feature_names = feature_names[:-1]

    return X, y, X_feature_names
