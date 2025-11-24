from sklearn.metrics import root_mean_squared_error
import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns:
        float: The RMSE value.
    """
    return root_mean_squared_error(y_true, y_pred)
