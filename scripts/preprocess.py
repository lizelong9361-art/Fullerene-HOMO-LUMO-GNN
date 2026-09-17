"""Build a validated graph cache from the 7,035 CSV/GJF sample pairs."""

from __future__ import annotations

import argparse

from src.dataset import load_or_build, manifest_records, repository_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = repository_root()
    if args.validate_only:
        frame, _ = manifest_records(root)
        print(f"Validated {len(frame)} labels and structure paths")
        return
    data = load_or_build(root, rebuild=args.rebuild, workers=args.workers)
    print(f"Processed {len(data)} graphs into results/processed_data.pt")


if __name__ == "__main__":
    main()
