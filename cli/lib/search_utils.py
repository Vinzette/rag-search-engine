import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT/'data'
MOVIES_PATH = DATA_PATH/'movies.json'
STOPWORDS_PATH = DATA_PATH/'stopwords.txt'
CACHE_PATH = PROJECT_ROOT/'cache'

def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
        return data['movies']

def load_stopwords():
    with open(STOPWORDS_PATH, "r") as f:
        data = f.read().splitlines()
    return data



