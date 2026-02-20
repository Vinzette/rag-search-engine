from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.semantic_search import cosine_similarity
from lib.search_utils import load_movies

class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_fpath):
        img = Image.open(image_fpath)
        return self.model.encode([img])[0]
    
    def search_with_image(self, image_fpath, limit=5):
        image_emb = self.embed_image(image_fpath)

        similarities = []
        for idx, text_emb in enumerate(self.text_embeddings):
            similarities.append((idx, cosine_similarity(image_emb, text_emb)))
        sorted_sims = sorted(similarities, key=lambda x:x[1], reverse=True)
        sorted_sims = sorted_sims[:limit]
        results = []

        for (idx,score) in sorted_sims:
            _doc=self.documents[idx]
            results.append({
                'title': _doc['title'],
                'description': _doc['description'],
                'doc_id': idx,
                'score': score
            })
        return results

def image_search_command(image_fpath, limit=5):
    movies = load_movies()
    ms = MultimodalSearch(movies)
    res = ms.search_with_image(image_fpath, limit)
    for i, r in enumerate(res, start=1):
        print(f"{i}. {r['title']} (similarity: {r['score']:.3f})")
        print(f"    {r['description'][:100]}")



def verify_image_embedding(image_fpath):
    ms = MultimodalSearch()
    embedding = ms.embed_image(image_fpath)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
