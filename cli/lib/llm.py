import os
from dotenv import load_dotenv
from google import genai
import json
from lib.search_utils import PROMPT_PATH

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

model='gemini-2.5-flash'
client = genai.Client(api_key=api_key)

def augment_prompt(query, type):
    with open(PROMPT_PATH/f'{type}.md', 'r') as f:
        prompt = f.read()
    return  generate_content(prompt, query)

def generate_content(prompt, query, **kwargs):
    prompt = prompt.format(query=query, **kwargs)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

# def correct_spelling(query):
#     return augment_prompt(query, 'spelling')

# def rewrite_query(query):
#     return augment_prompt(query, 'rewrite')

# def expand_query(query):
#     return augment_prompt(query, 'expand')

def llm_judge(query, formatted_results):
    with open(PROMPT_PATH/'llm_judge.md', 'r') as f:
        prompt = f.read()
    results =  generate_content(prompt, query,formatted_results=formatted_results)
    results = json.loads(results)
    return results

def answer_question(query, documents):
    with open(PROMPT_PATH/'answer_question.md', 'r') as f:
        prompt = f.read()
    results = generate_content(prompt, query=query, docs=documents)
    return results


    
