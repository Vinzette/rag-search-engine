import os
from lib.llm import correct_spelling, rewrite_query, expand_query
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from lib.search_utils import load_movies

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit*500)
        sem_results = self.semantic_search.search_chunks(query, limit*500) #cutting computation by searching not all docs
        combined_results = combine_search_results(bm25_results, sem_results,alpha)
        return combined_results
    
    def rrf_search(self, query, k, limit=10):
        bm25_results = self._bm25_search(query, limit*500)
        sem_results = self.semantic_search.search_chunks(query, limit*500) #cutting computation by searching not all docs
        combined_results = rrf_combine_search_results(bm25_results, sem_results, k)
        return combined_results

    
def rrf_search(query, k=60, limit=5, enhance=None):
     movies = load_movies()
     hs = HybridSearch(movies)
     results = hs.rrf_search(query, k, limit)
     match enhance:
          case "spell":
               new_query=correct_spelling(query)
               print(f"Enhanced query (spell): '{query}' -> '{new_query}'\n")
               query = new_query
          case "rewrite":
               new_query = rewrite_query(query)
               print(f"Enhanced query (rewrite): '{query}' -> '{new_query}'\n")
               query = new_query
          case "expand":
               new_query = expand_query(query)
               print(f"Enhanced query (expand): '{query}' -> '{new_query}'\n")
               query = new_query 

     results = hs.rrf_search(query, k, limit)
     for idx, r in enumerate(results[:limit], start =1):
           print(f"{idx}. {r['title']}")
           print(f"RRF Score Score: {r['rrf_score']}")
           print(f"BM25 Rank: {r['bm25_rank']}, Semantic Rank: {r['sem_rank']}")
           print(r['description'][:100])

def rrf_score(rank, k):
     return 1 / (rank + k)

def rr_final_score(r1,r2,k):
     if r1 and r2:
          return rrf_score(r1, k) + rrf_score(r2, k)
     return 0.

def rrf_combine_search_results(bm25_results, sem_results,k):
     scores = {}
     for rank, result in enumerate(bm25_results, start=1):
          doc_id = result['doc_id']
          scores[doc_id] = {
               'bm25_rank': rank,
               'bm25_score': rrf_score(rank, k),
               'sem_rank': None,
               'sem_score': None,
               'title': result['title'],
               'description' : result['description']
          }
     for rank, result in enumerate(sem_results, start=1):
          doc_id = result['id']
          if doc_id not in scores:
               scores[doc_id] = {
               'doc_id': doc_id,
               'bm25_rank': None,
               'bm25_score': None,
               'sem_rank': None,
               'sem_score': None,
               'title': result['title'],
               'description' : result['document']
               }
          scores[doc_id]['sem_rank']= rank
          scores[doc_id]['sem_score'] = rrf_score(rank, k)
     
     for doc_id in scores.keys():
          scores[doc_id]['rrf_score'] = rr_final_score(
               scores[doc_id]['bm25_rank'],
               scores[doc_id]['sem_rank'],
               k
          )
     results = sorted(list(scores.values()), key = lambda x: x['rrf_score'], reverse=True)
     return results


def weighted_search(query, alpha, limit):
     movies = load_movies()
     hs = HybridSearch(movies)
     results = hs.weighted_search(query, alpha=0.5, limit=5)
     for idx, r in enumerate(results[:limit]):
           print(f"{idx+1} {r['title']}")
           print(f"Hybrid Score: {r['hybrid_score']}")
           print(f"BM25: {r['bm25_score']}, Semantic: {r['sem_score']}")
           print(r['description'][:100])
    
def hybrid_score(bm25_score, sem_score, alpha=0.5):
     return(alpha* bm25_score) + ((1-alpha) * sem_score)

def normalize_search_results(results):
     scores =  [r['score'] for r in results]
     norm_scores = normalize_scores(scores)
     for idx, result in enumerate(results):
          result['normalized_score'] = norm_scores[idx]
     return results

def combine_search_results(bm25_results, sem_results, alpha):
     bm25_norm = normalize_search_results(bm25_results)
     sem_norm = normalize_search_results(sem_results)

     combined_norm = {}
     for norm in bm25_norm:
          doc_id = norm['doc_id']
          combined_norm[doc_id] = {
               'doc_id': doc_id,
               'bm25_score': norm['normalized_score'],
               'sem_score': 0.,
               'title': norm['title'],
               'description' : norm['description']
               }
     for norm in sem_norm:
        doc_id = norm['id']
        if doc_id not in combined_norm:
             combined_norm[doc_id] = {
               'doc_id': doc_id,
               'bm25_score': 0.,
               'sem_score': 0.,
               'title': norm['title'],
               'description' : norm['document']
            }
        combined_norm[doc_id]['sem_score'] = norm['normalized_score']

     for k,v in combined_norm.items():
          combined_norm[k]['hybrid_score'] = hybrid_score(v['bm25_score'], v['sem_score'], alpha)

     results = sorted(list(combined_norm.values()), key= lambda x:x['hybrid_score'], reverse=True)
     return results

             

def normalize_scores(scores):
        if not scores : 
             return[]
        min_score = min(scores)
        max_score = max(scores)

        if min_score == max_score: 
             return [1.]*len(scores) # if all scores are the same, return 1 for all to avoid division by zero and maintain relative ranking

        score_range = max_score - min_score
        return [(score - min_score)/score_range for score in scores] 