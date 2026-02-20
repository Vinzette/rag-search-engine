import argparse
import mimetypes
from google.genai import types
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

model='gemini-2.5-flash'
client = genai.Client(api_key=api_key)
 
system_prompt = '''Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary'''

def main() -> None:
    parser = argparse.ArgumentParser(description="Describe Image")
    
    parser.add_argument("--image", type=str, help="Path to image for the rewriting" )
    parser.add_argument("--query", type=str, help = "User query that goes with the image")
    
    args = parser.parse_args()

    mime,_=mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, 'rb') as f:
        img = f.read()
    
    parts = [
    system_prompt,
    types.Part.from_bytes(data=img, mime_type=mime),
    args.query.strip(),
]
    response = client.models.generate_content(model=model, contents = parts)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

if __name__=='__main__':
    main()