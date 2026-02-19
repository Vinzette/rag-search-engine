import argparse
from lib.hybrid_search import normalize_scores, weighted_search, rrf_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    norm_parser = subparsers.add_parser("normalize", help="Normalize of list of scores")
    norm_parser.add_argument("scores", type=float, nargs='+', help="List of scores to normalize")

    ws_parser = subparsers.add_parser("weighted-search", help="A hybrid search with weighted average combination")
    ws_parser.add_argument("query", type=str, help="User query to find relevant docs for")
    ws_parser.add_argument("--alpha", type=float, default = 0.5, help="percent of weight for BM25")
    ws_parser.add_argument("--limit", type=int, default = 5, help="number of results to return")

    rrf_parser = subparsers.add_parser("rrf-search", help="A Hybrid search with rrf combination")
    rrf_parser.add_argument("query", type=str, help="User query to find relevant docs for")
    rrf_parser.add_argument("--k", type=int, default =60, help="k hyperparam for rrf")
    rrf_parser.add_argument("--limit", type=int, default = 5, help="number of results to return")
    rrf_parser.add_argument("--enhance", type=str, choices=["spell","rewrite","expand"], help="Query enhancement method")
    rrf_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Rerank Method")
    rrf_parser.add_argument("--debug", type=str, help="A piece of title to track")

    args = parser.parse_args()

    match args.command:
        case "rrf-search":
            rrf_search(args.query, args.k, args.limit, args.enhance, args.rerank_method, args.debug)

        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)

        case "normalize":
            norm_scores = normalize_scores(args.scores)
            for norm_score in norm_scores:
                print(f"* {norm_score:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()