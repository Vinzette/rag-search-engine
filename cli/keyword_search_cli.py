#!/usr/bin/env python3
from lib.keyword_search import (search_command, build_command, tf_command, idf_command, tfidf_command) 
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    search_parser = subparsers.add_parser("build", help="Build pkl file")

    search_parser = subparsers.add_parser("tf", help="Calculate TF")
    search_parser.add_argument("doc_id", type=int, help="Doc ID for check")
    search_parser.add_argument("term", type=str, help="Search term to find counts for")
    
    search_parser = subparsers.add_parser("idf", help="Calculate IDF")
    search_parser.add_argument("term", type=str, help="Search term to find ID for")

    search_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF")
    search_parser.add_argument("doc_id", type=int, help="Doc ID for check")
    search_parser.add_argument("term", type=str, help="Search term to find TF-IDF for")


    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            results=search_command(args.query, 5)
            for i, result in enumerate(results):
                print(f"{i} {result['title']}")
        case "build":
            build_command()
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()