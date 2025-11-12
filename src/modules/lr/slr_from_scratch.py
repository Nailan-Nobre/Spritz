from typing import Any


class SimpleLinearRegressionFS:
    def __init__(self) -> None:
        """Classe que implementa minha regressão simples"""
        self.__b0 = None
        self.__b1 = None

    def train(self) -> Any:
        """Método para treinar o modelo, usa o método Ordinary Least Squares

        Returns:
            Any: _description_
        """
        # Implementar o OLS aqui!!
        return None

    def get_b0(self) -> Any:
        return self.__b0

    def get_b1(self) -> Any:
        return self.__b1
