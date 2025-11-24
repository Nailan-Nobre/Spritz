from modules.loaders.music import fetch_data
from modules.lr.slr import LR
from modules.metrics.reg.rmse import rmse


def main():
    source = "data/spotify_songs.csv"
    data = fetch_data(source)
    model = LR()
    model.train(data[0], data[1])
    print("Score:", model.get_score(data[0], data[1]))
    print("Predictions:", model.predict(data[0]))
    print("Intercept:", model.get_intercept())
    print("Coefficients:", model.get_coefficients())
    y_pred = model.predict(data[0])
    print("predict values:", y_pred)
    print("RMSE:", rmse(data[1], y_pred))


if __name__ == "__main__":
    main()
