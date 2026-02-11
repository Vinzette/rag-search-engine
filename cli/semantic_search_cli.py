#!/usr/bin/env python3

import argparse
from lib.semantic_search import (verify_model, embed_text)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("verify", help="Verify the Embedding Model loads properly")

    embed_parser = subparsers.add_parser("embed_text", help="Encode Text using embedding model")
    embed_parser.add_argument("text", type=str, help="Text to be encoded")
    
    args = parser.parse_args()

    match args.command:
        case "embed_text":
            embed_text(args.text)

        case "verify":
            verify_model()
        
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()