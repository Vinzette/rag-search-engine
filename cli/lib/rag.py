from lib.llm import answer_question, summarize_documents
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

def query_answering(query):
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query, k=60, limit=5)
    print("Search Results:")
    for res in rrf_results:
        print(f" - {res['title']}")
    rag_results = answer_question(query,rrf_results)
    print("RAG Response:")
    print(rag_results)

def doc_summarization(query, limit=5):
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query, k=60, limit=limit)
    print("Search Results:")
    for res in rrf_results:
        print(f" - {res['title']}")
    rag_results = answer_question(query,rrf_results)
    print("LLM Summary:")
    print(rag_results)




