import os
import time
from dotenv import load_dotenv
from google import genai
from lib.search_utils import PROMPT_PATH

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

model='gemini-2.5-flash'
client = genai.Client(api_key=api_key)

def individual_rerank(query, documents):
    with open(PROMPT_PATH/'individual_rerank.md', 'r') as f:
        prompt = f.read()
    results = []
    for doc in documents:
        _prompt = prompt.format(
            query=query, 
            title=doc['title'], 
            description=doc['description'])
        #response = client.models.generate_content(model=model, contents=_prompt)
        try:
            response = client.models.generate_content(model=model, contents=_prompt)
        except Exception:
            results.append({**doc, "rerank_response": 0})
            continue

        clean_response_text = (response.text or "").strip()
        try:
            clean_response_text = int(clean_response_text)
        except:
            print(f"Failed to case {response.text} to int for  {doc['title']}")
            clean_response_text = 0
        results.append({**doc, "rerank_response":clean_response_text})
        #time.sleep(3)
        # except Exception as e:
        #     print((clean_response_text,response.text, query, doc['description'][:100]))
        #     raise e
    results = sorted(results, key=lambda x: x['rerank_response'], reverse=True)
    return results