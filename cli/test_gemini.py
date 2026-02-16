import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")


model='gemini-2.5-flash'
client = genai.Client(api_key=api_key)


def generate_content():
    prompt = "Why is Gooseberry Icecream from naturals so speacial? Generate one paragraph answer"
    response = client.models.generate_content(model=model, contents=prompt)
    print(response.text)
    print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")

if __name__=='__main__':
    generate_content()