from modules.loaders.music import feth_data


def main():
    source = "data/music_data.csv"
    data = feth_data(source)
    print(type(data))


if __name__ == "__main__":
    main()
