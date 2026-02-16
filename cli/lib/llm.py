import os
from dotenv import load_dotenv
from google import genai
from lib.search_utils import PROMPT_PATH

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

print("Using GEMINI_API_KEY from environment")


model='gemini-2.5-flash'
client = genai.Client(api_key=api_key)

def augument_prompt(query, type):
    with open(PROMPT_PATH/f'{type}.md', 'r') as f:
        prompt = f.read()
    return  generate_content(prompt, query)

def generate_content(prompt, query):
    prompt = prompt.format(query=query)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

def correct_spelling(query):
    return augument_prompt(query, 'spelling')

def rewrite_query(query):
    return augument_prompt(query, 'rewrite')

def expand_query(query):
    return augument_prompt(query, 'expand')


    
