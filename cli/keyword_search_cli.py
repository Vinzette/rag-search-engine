#!/usr/bin/env python3
from lib.keyword_search import search_command
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            results=search_command(args.query, 5)
            for i, result in enumerate(results):
                print(f"{i} {result['title']}")

            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()