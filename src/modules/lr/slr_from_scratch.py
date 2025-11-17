"""
Módulo: SimpleLinearRegression
Implementa uma regressão linear simples (OLS) sem bibliotecas externas.
"""


class SimpleLinearRegression:
    """
    Classe que representa um modelo de Regressão Linear Simples.

    A equação ajustada é:
        y = b0 + b1 * x
    onde:
        b0 = intercepto
        b1 = inclinação (coeficiente angular)
    """

    def __init__(self):
        """Inicializa os parâmetros do modelo."""
        self.b0 = 0  # Intercepto
        self.b1 = 0  # Inclinação

    def fit(self, X, y):
        """
        Ajusta o modelo aos dados usando o método dos Mínimos Quadrados Ordinários (OLS).
        Args:
            X (list[float]): Lista com valores da variável independente (x)
            y (list[float]): Lista com valores da variável dependente (y)
        """
        n = len(X)
        media_x = sum(X) / n
        media_y = sum(y) / n

        numerador = 0
        denominador = 0
        for i in range(n):
            numerador += (X[i] - media_x) * (y[i] - media_y)
            denominador += (X[i] - media_x) ** 2

        self.b1 = numerador / denominador
        self.b0 = media_y - self.b1 * media_x

    def predict(self, X):
        """
        Retorna previsões para os valores de X fornecidos.
        Args:
            X (list[float]): Lista com os valores de entrada.
        Returns:
            list[float]: Valores previstos de y.
        """
        return [self.b0 + self.b1 * x for x in X]

    def r2_score(self, y_true, y_pred):
        """
        Calcula o coeficiente de determinação (R²) do modelo.
        Args:
            y_true (list[float]): Valores reais de y.
            y_pred (list[float]): Valores previstos de y.
        Returns:
            float: Valor de R² entre 0 e 1.
        """
        media_y = sum(y_true) / len(y_true)
        ss_total = sum((yi - media_y) ** 2 for yi in y_true)
        ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
        return 1 - (ss_res / ss_total)
