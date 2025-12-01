from modules.loaders.music import fetch_data
from modules.lr.slr import LR
from modules.metrics.reg.rmse import rmse


def main():
    print("=" * 70)
    print("ANÁLISE DE REGRESSÃO LINEAR - SPOTIFY SONGS")
    print("=" * 70)

    # Carregamento dos dados
    source = "data/spotify_songs.csv"
    print(f"\n📂 Carregando dados de: {source}")
    X, y, feature_names = fetch_data(source)
    print("✓ Dados carregados com sucesso!")
    print(f"   - Total de amostras: {len(X)}")
    print(f"   - Número de features: {X.shape[1]}")

    # Treinamento do modelo
    print("\n🤖 Treinando modelo de Regressão Linear...")
    model = LR()
    model.train(X, y)
    print("✓ Modelo treinado com sucesso!")

    # Avaliação do modelo
    print("\n" + "=" * 70)
    print("RESULTADOS DO MODELO")
    print("=" * 70)

    score = model.get_score(X, y)
    print(f"\n📊 R² Score (Coeficiente de Determinação): {score:.4f}")
    print(f"   → O modelo explica {score * 100:.2f}% da variância dos dados")

    intercept = model.get_intercept()
    print(f"\n📐 Intercepto (b0): {intercept:.2f}")

    coefficients = model.get_coefficients()
    print("\n📈 Coeficientes (pesos das features):")
    for i, (name, coef) in enumerate(zip(feature_names, coefficients), 1):
        print(f"   {i:2d}. {name:20s}: {coef:12.2f}")

    # Predições
    y_pred = model.predict(X)
    print(f"\n🔮 Predições geradas: {len(y_pred)} valores")
    print(f"   Exemplo das primeiras 5 predições: {y_pred[:5]}")

    # Erro do modelo
    error = rmse(y, y_pred)
    print(f"\n📉 RMSE (Root Mean Squared Error): {error:.2f}")
    print("   → Erro médio das predições")

    print("\n" + "=" * 70)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
