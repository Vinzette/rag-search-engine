#!/usr/bin/env python3
from lib.keyword_search import (
    search_command, build_command, tf_command, idf_command, tfidf_command, bm25_idf_command, bm_25_tf_command, bm25_search) 
import argparse
from lib.search_utils import BM25_K1, BM25_B


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

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 K1 parameter")

    
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=float, nargs='?', default = 5, help="Top N Results")

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
        case "bm25tf":
            bm_25_tf_command(args.doc_id, args.term, args.k1, args.b)
        
        case "bm25idf":
            bm25_idf_command(args.term)
        
        case "bm25search":
            bm25_results = bm25_search(args.query, args.limit)
            for idx, res in enumerate(bm25_results):
                print(f"{idx}. {res['doc_id']} {res['title']} - Score: {res['score']:.2f}")
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()