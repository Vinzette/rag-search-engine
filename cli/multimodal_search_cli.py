import argparse
from lib.multimodal_search import verify_image_embedding

def main() -> None:
    parser = argparse.ArgumentParser(description="Describe Image")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(name="verify_image_embedding", help="Verify image embeddings load")
    verify_parser.add_argument('image_fpath', type=str, help="path to the image to process")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_fpath)

if __name__=='__main__':
    main()