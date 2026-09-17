"""Train using the published fixed random 8:1:1 split for one seed."""

from src.train import main


if __name__ == "__main__":
    main(protocol_default="random")
