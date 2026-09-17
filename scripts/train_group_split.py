"""Train using the published C20-C56 / C58-C60 / C70-C100 split."""

from src.train import main


if __name__ == "__main__":
    main(protocol_default="group")
