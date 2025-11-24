import numpy as np
import pandas as pd


def fetch_data(source: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Fetch data from the given source.

    Args:
        source (str): The data source URL or file path.

    Returns:
        np.ndarray: An array of data records.
    """
    df = pd.read_csv(source).to_numpy()
    X = df[:, :-1]
    y = df[:, -1]
    return X, y
