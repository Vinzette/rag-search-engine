import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Describe Image")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(name="image_search", help="Search for an movie based on an image")
    verify_parser.add_argument('image_fpath', type=str, help="path to the image to search with")
    verify_parser.add_argument('--limit', type=int, default=5, help="Number of results to return")


    args = parser.parse_args()

    match args.command:
        case "image_search":
            image_search_command(args.image_fpath, args.limit)
        case "verify_image_embedding":
            verify_image_embedding(args.image_fpath)

if __name__=='__main__':
    main()