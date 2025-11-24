from sklearn.linear_model import LinearRegression
import numpy as np


class LR:
    def __init__(self):
        self.__model = LinearRegression()

    def train(self, X: np.ndarray, y: np.ndarray):
        self.__model.fit(X, y)

    def get_score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.__model.score(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.__model.predict(X)

    def get_intercept(self) -> float:
        return self.__model.intercept_

    def get_coefficients(self) -> np.ndarray:
        return self.__model.coef_


if __name__ == "__main__":
    # Example usage
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([3, 5, 7])

    model = LR()
    model.train(X, y)

    print("Score:", model.get_score(X, y))
    print("Predictions:", model.predict(X))
    print("Intercept:", model.get_intercept())
    print("Coefficients:", model.get_coefficients())
