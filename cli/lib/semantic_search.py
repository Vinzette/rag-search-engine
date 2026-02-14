from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
from pathlib import Path
from lib.search_utils import load_movies
import re
import json
from collections import defaultdict

load_dotenv()


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        # self.documents: A list of movie dictionaries (id, title, description) 
        # where the index matches the order of the vectors in self.embeddings.

        # self.document_map: A dictionary that maps movie IDs to their movie 
        # data for fast lookups when you only have a specific ID.
        self.embeddings_path = Path("cache/movie_embeddings.npy")
    
    def build_embeddings(self, documents):
        self.document_map = {}
        self.documents = documents
        movie_strings= []
        for doc in self.documents:
            self.document_map[doc['id']] = doc #in the document map, put the doc itself as the value, and the id as the key
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            if len(self.documents) == len(self.embeddings):
                return self.embeddings
        return self.build_embeddings(documents)    


    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Must have text to create an embedding")
        return self.model.encode([text])[0] #encode method expects a list of inputs, but we're only passing in one, so we wrap it in a list. It also returns a corresponding list as output, but we only care about the first element because we're only passing in one input.
    
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        qry_emb = self.generate_embedding(query)

        similarities = []
        for doc_emb, doc in zip(self.embeddings, self.documents):
            _similarity = cosine_similarity(qry_emb, doc_emb)
            similarities.append((_similarity, doc))
        similarities.sort(key=lambda x: x[0], reverse=True)
        res = []
        for sc, doc in similarities[:limit]:
            res.append({'score':sc, 
                        'title':doc['title'], 
                        'description':doc['description']})
        return res

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_embeddings_path = Path('cache/chunk_embeddings.npy')
        self.chunk_metadata = None
        self.chunk_metadata_path = Path('cache/chunk_metadata.json')    
    
    def build_chunk_embeddings(self, documents):
        self.documents = documents
        # Create a hash map for O(1) lookups of documents by ID
        self.document_map = {doc['id']: doc for doc in documents} #loop thru documents and for each movie, grab the ID
        
        all_chunks = []
        chunk_metadata = []
        for midx, doc in enumerate(documents):
            if doc['description'].strip() == '':
                continue
            _chunks = semantic_chunking(doc['description'], overlap=1, max_chunk_size=4)
            all_chunks += _chunks
            for cidx in range(len(_chunks)):
                chunk_metadata.append({"movie_idx": midx, "chunk_idx": cidx,  "total_chunks": len(_chunks) })
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        self.chunk_embeddings_path.parent.mkdir(parents = True, exist_ok=True) #make sure the cache directory exists before we try to save the file, if it doesn't exist, it will create it. parents=True allows it to create any necessary parent directories as well, and exist_ok=True means that if the directory already exists, it won't raise an error.
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, 'w') as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        
        if self.chunk_embeddings_path.exists() and self.chunk_metadata_path.exists():
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, 'r') as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10):
        query_emb = self.generate_embedding(query)
        chunk_scores = []
        movie_scores = defaultdict(lambda: 0) #lambda is just a way to create a default value for any key that doesn't exist in the dictionary, in this case we want the default score to be 0, so if we try to access a key that hasn't been added to the movie_scores dictionary yet, it will return 0 instead of throwing an error. This is useful because when we calculate the similarity score for each chunk, we want to keep track of the highest score for each movie, and if we haven't seen that movie before, we want its initial score to be 0. 
        for idx in range(len(self.chunk_embeddings)):
            chunk_embedding = self.chunk_embeddings[idx]
            metadata = self.chunk_metadata['chunks'][idx]
            midx, cidx = metadata['movie_idx'], metadata['chunk_idx']
            sim = cosine_similarity(query_emb, chunk_embedding)
            chunk_scores.append({
                'movie_idx': midx,
                'chunk_idx': cidx,
                'score': sim})
            movie_scores[midx] = max(movie_scores[midx], sim)
        movie_scores_sorted = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True) #changes dict into tuples, [1:2] --> [(1,2)] then  we can sort 
        res = []
        for midx, score in movie_scores_sorted[:limit]:
            doc = self.documents[midx]
            res.append({
                "id": doc['id'],
                "title": doc['title'],
                "document": doc['description'][:100],
                "score": round(score, 4),
                "metadata": {}})
        return res

def search_chunked(query, limit=5):
    css = ChunkedSemanticSearch()
    movies = load_movies()
    _ = css.load_or_create_chunk_embeddings(movies)
    results = css.search_chunks(query, limit)
    for i, res in enumerate(results):
        print(f"\n{i+1}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['document']}...")

def embed_chunks():
    movies = load_movies()
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")



def semantic_chunking(text, max_chunk_size=4, overlap=0):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    step_size = max_chunk_size - overlap
    for i in range(0, len(sentences), step_size):
        chunk_sentences = sentences[i:i+max_chunk_size]
        if len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
    return chunks

def chunk_text_semantic(text, max_chunk_size=4, overlap=0):
    chunks = semantic_chunking(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")

def fixed_sized_chunking(text, overlap=0, chunk_size=200):
    words = text.split()
    chunks = []
    step_size = chunk_size - overlap
    for i in range(0, len(words), step_size): #it will go from 0 to whatever the chunk size is
        chunk_words = words[i:i+chunk_size]
        if len(chunk_words) <= overlap: #if last chunk is smaller than the overlap, we can just break
            break
        chunks.append(" ".join(chunk_words))
    return chunks

def chunk_text(text, overlap=0, chunk_size=200):
    chunks = fixed_sized_chunking(text, overlap, chunk_size)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")

def search(query, limit=5):
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    search_results = ss.search(query, limit)
    for idx, res in enumerate(search_results):
        print(f"{idx}. {res['title']} (score: {res['score']:.4f})")
        print(res['description'][:100])

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")         

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")