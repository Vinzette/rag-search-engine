import argparse
from lib.hybrid_search import normalize_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    norm_parser = subparsers.add_parser("normalize", help="Available commands")
    norm_parser.add_argument("scores", type=float, nargs='+', help="List of scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            norm_scores = normalize_scores(args.scores)
            for norm_score in norm_scores:
                print(f"* {norm_score:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()