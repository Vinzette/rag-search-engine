# 🔎 RAG Search Engine

A movie search engine that walks through the full evolution of information retrieval — from plain keyword matching all the way to retrieval-augmented generation with an LLM.

This project started as a collection of CLI experiments and grew into a complete search pipeline backed by a Streamlit web app. The dataset is **5,000 movies** with titles and descriptions.

The pipeline covers:
- **BM25** — classic term-frequency keyword search with an inverted index
- **Semantic Search** — dense vector retrieval using sentence embeddings
- **Hybrid Search** — combining both via Reciprocal Rank Fusion (RRF) or weighted score normalization
- **Query Enhancement** — spelling correction, query rewriting, and query expansion via Gemini
- **Reranking** — LLM-based individual/batch reranking and cross-encoder reranking
- **RAG** — retrieving relevant documents then generating natural language answers, summaries, and citations with Gemini
- **Multimodal Search** — finding movies from images using CLIP embeddings
- **Evaluation** — precision@k, recall@k, and F1 against a hand-built golden dataset

---

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (for dependency management & running scripts)
- A [Google Gemini API key](https://aistudio.google.com/apikey) (needed for RAG, query enhancement, reranking, and image description)

### Installation

```bash
git clone https://github.com/Vinzette/rag-search-engine.git
cd rag-search-engine
uv sync
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

### Download the dataset

The `data/` directory (movies, golden dataset, stopwords) is not checked into git. Download it from the latest GitHub release:

```bash
bash download_data.sh
```

Or grab it manually from the [Releases page](https://github.com/Vinzette/rag-search-engine/releases).

### First-time setup

Build the inverted index (BM25) so keyword search works out of the box:

```bash
uv run cli/keyword_search_cli.py build
```

Embedding caches (`cache/movie_embeddings.npy`, `cache/chunk_embeddings.npy`, etc.) are generated automatically the first time you run a semantic or hybrid search.

---

## Web Interface

The easiest way to use everything is through the Streamlit app:

```bash
uv run streamlit run app.py
```

It opens at `http://localhost:8501` with six pages accessible from the sidebar:

| Page | What it does |
|---|---|
| **Keyword Search** | BM25 search, simple keyword search, and low-level scoring tools (TF, IDF, TF-IDF, BM25 TF/IDF) |
| **Semantic Search** | Full-doc and chunked vector search, text/query embedding inspection, chunking demos, model verification |
| **Hybrid Search** | RRF and weighted hybrid search with optional query enhancement, reranking, debug tracking, and LLM-as-judge evaluation |
| **RAG (Q&A)** | Ask questions, get summaries, citations, or detailed answers generated from retrieved movie documents |
| **Image / Multimodal** | Search by uploading an image (CLIP), or use Gemini to rewrite a text query based on an image |
| **Evaluation** | Run precision@k, recall@k, and F1 benchmarks against the golden dataset |

---

## CLI Tools

Every feature is also available as a standalone CLI command. These are useful for scripting, debugging, and understanding each piece of the pipeline in isolation.

All commands are run from the project root with `uv run`:

### 1. Keyword Search — `cli/keyword_search_cli.py`

The classic information retrieval stack: inverted index, TF-IDF, and BM25.

```bash
# Build the inverted index (run this first)
uv run cli/keyword_search_cli.py build

# Simple keyword search
uv run cli/keyword_search_cli.py search "space adventure"

# Full BM25 ranked search
uv run cli/keyword_search_cli.py bm25search "science fiction aliens" 10

# Inspect scoring internals
uv run cli/keyword_search_cli.py tf 42 "robot"          # term frequency
uv run cli/keyword_search_cli.py idf "robot"             # inverse document frequency
uv run cli/keyword_search_cli.py tfidf 42 "robot"        # TF-IDF
uv run cli/keyword_search_cli.py bm25tf 42 "robot" 1.5 0.75  # BM25 TF (with k1, b)
uv run cli/keyword_search_cli.py bm25idf "robot"         # BM25 IDF
```

### 2. Semantic Search — `cli/semantic_search_cli.py`

Dense vector retrieval with sentence-transformers. Supports both full-document and chunked embeddings.

```bash
# Verify the embedding model loads
uv run cli/semantic_search_cli.py verify

# Embed a piece of text and inspect the vector
uv run cli/semantic_search_cli.py embed_text "A story about a lost astronaut"

# Embed a search query
uv run cli/semantic_search_cli.py embedquery "time travel movies"

# Full-document semantic search
uv run cli/semantic_search_cli.py search "movies about time travel" --limit 5

# Chunked semantic search (better for long descriptions)
uv run cli/semantic_search_cli.py search_chunked "movies about time travel" --limit 5

# Try different chunking strategies
uv run cli/semantic_search_cli.py chunk "Your long text here..." --chunk-size 200 --overlap 20
uv run cli/semantic_search_cli.py semantic_chunk "Your long text here..." --max-chunk-size 4 --overlap 1

# Build/load chunk embeddings for all movies
uv run cli/semantic_search_cli.py embed_chunks

# Verify saved embeddings match the dataset
uv run cli/semantic_search_cli.py verify_embeddings
```

### 3. Hybrid Search — `cli/hybrid_search_cli.py`

Combines BM25 and semantic search. Two fusion strategies: Reciprocal Rank Fusion (RRF) and weighted score normalization.

```bash
# RRF hybrid search
uv run cli/hybrid_search_cli.py rrf-search "funny action movies" --k 60 --limit 5

# Weighted hybrid search (alpha controls BM25 vs semantic weight)
uv run cli/hybrid_search_cli.py weighted-search "dark knight" --alpha 0.5 --limit 5

# Query enhancement before searching
uv run cli/hybrid_search_cli.py rrf-search "scary movie" --enhance rewrite
uv run cli/hybrid_search_cli.py rrf-search "scary movie" --enhance expand
uv run cli/hybrid_search_cli.py rrf-search "scary movie" --enhance spell

# Reranking the results
uv run cli/hybrid_search_cli.py rrf-search "scary movie" --rerank-method cross_encoder
uv run cli/hybrid_search_cli.py rrf-search "scary movie" --rerank-method individual
uv run cli/hybrid_search_cli.py rrf-search "scary movie" --rerank-method batch

# Debug: track where a specific movie lands in the pipeline
uv run cli/hybrid_search_cli.py rrf-search "bear movies" --debug "Paddington" --rerank-method cross_encoder

# LLM-as-judge: have Gemini rate each result's relevance
uv run cli/hybrid_search_cli.py rrf-search "bear movies" --evaluate

# Normalize a list of scores (utility)
uv run cli/hybrid_search_cli.py normalize 0.5 1.2 3.7 0.1
```

### 4. RAG & Q/A — `cli/augmented_generation_cli.py`

Retrieves relevant movies then uses Gemini to answer questions, summarize, or cite sources.

```bash
# Simple question answering
uv run cli/augmented_generation_cli.py rag "Who directed Inception and what is it about?"

# Summarize retrieved documents
uv run cli/augmented_generation_cli.py summarize "The Matrix" --limit 5

# Answer with citations referencing specific movies
uv run cli/augmented_generation_cli.py citations "List movies starring Tom Hanks" --limit 5

# Detailed question answering
uv run cli/augmented_generation_cli.py question "Compare the themes of Interstellar and Gravity" --limit 5
```

### 5. Multimodal Search — `cli/multimodal_search_cli.py`

Uses CLIP to search for movies by image similarity.

```bash
uv run cli/multimodal_search_cli.py image_search path/to/image.jpg --limit 5
```

### 6. Image-to-Query Rewriting — `cli/describe_image_cli.py`

Sends an image + text query to Gemini, which rewrites the query to incorporate what it sees in the image — useful for improving search when you have a movie poster or screenshot.

```bash
uv run cli/describe_image_cli.py --image path/to/poster.jpg --query "find similar movies"
```

### 7. Evaluation — `cli/evaluation_cli.py`

Benchmarks hybrid search against `data/golden_dataset.json` using Precision@k, Recall@k, and F1.

```bash
uv run cli/evaluation_cli.py --limit 5
```

---

## How the Pipeline Works

```
User Query
    │
    ├─ [Optional] Query Enhancement (Gemini: spelling / rewrite / expand)
    │
    ├─────────────────┬──────────────────┐
    │                 │                  │
    ▼                 ▼                  ▼
  BM25           Semantic           Image (CLIP)
 (Inverted       (Chunked           Embeddings
  Index)          Embeddings)
    │                 │                  │
    └────────┬────────┘                  │
             ▼                           │
       Hybrid Fusion                     │
       (RRF or Weighted)                 │
             │                           │
             ├─ [Optional] Reranking ◄───┘
             │   (Cross-Encoder / LLM)
             │
             ├─ [Optional] LLM-as-Judge Evaluation
             │
             ▼
       Retrieved Documents
             │
             ▼
       RAG Generation (Gemini)
       ├─ Answer question
       ├─ Summarize
       ├─ Answer with citations
       └─ Detailed Q&A
```

---

## Project Structure

```
rag-search-engine/
├── app.py                          # Streamlit web frontend
├── pyproject.toml                  # Dependencies & project metadata
├── .env                            # GEMINI_API_KEY (not committed)
│
├── cli/                            # CLI entry points (one per feature)
│   ├── keyword_search_cli.py       #   BM25 & keyword search
│   ├── semantic_search_cli.py      #   Embedding & vector search
│   ├── hybrid_search_cli.py        #   RRF / weighted fusion
│   ├── augmented_generation_cli.py #   RAG (Q&A, summarization, citations)
│   ├── multimodal_search_cli.py    #   CLIP image search
│   ├── describe_image_cli.py       #   Gemini image-to-query rewriting
│   ├── evaluation_cli.py           #   Precision/Recall/F1 benchmarks
│   │
│   └── lib/                        # Core logic (shared by CLI & Streamlit)
│       ├── keyword_search.py       #   Inverted index, tokenization, BM25
│       ├── semantic_search.py      #   Embeddings, chunking, cosine similarity
│       ├── hybrid_search.py        #   Score fusion (RRF, weighted, normalize)
│       ├── rag.py                  #   Orchestrates retrieval → generation
│       ├── llm.py                  #   Gemini API calls
│       ├── rerank.py               #   Individual, batch, & cross-encoder reranking
│       ├── multimodal_search.py    #   CLIP-based image search
│       ├── evaluation.py           #   Golden dataset benchmarking
│       ├── search_utils.py         #   Shared constants & data loading
│       └── prompts/                #   Prompt templates (Markdown files)
│           ├── answer_question.md
│           ├── answer_question_detailed.md
│           ├── answer_with_citations.md
│           ├── summarization.md
│           ├── spelling.md
│           ├── rewrite.md
│           ├── expand.md
│           ├── individual_rerank.md
│           ├── batch_rerank.md
│           └── llm_judge.md
│
├── data/
│   ├── movies.json                 # 5,000 movies (title + description)
│   ├── golden_dataset.json         # Hand-curated test cases for evaluation
│   └── stopwords.txt               # Stopword list for tokenization
│
└── cache/                          # Auto-generated (gitignored)
    ├── movie_embeddings.npy        # Full-document embeddings
    ├── chunk_embeddings.npy        # Chunked embeddings
    ├── chunk_metadata.json         # Chunk-to-movie mapping
    ├── index.pkl                   # BM25 inverted index
    ├── docmap.pkl                  # Document ID → movie mapping
    ├── term_frequencies.pkl        # Per-document term counts
    └── doc_lengths.pkl             # Document lengths for BM25 normalization
```

---

## Models & Tech

| Component | What's used |
|---|---|
| **LLM** | Google Gemini 2.5 Flash |
| **Sentence Embeddings** | `all-MiniLM-L6-v2` (via sentence-transformers) |
| **Multimodal Embeddings** | `clip-ViT-B-32` (via sentence-transformers) |
| **Cross-Encoder Reranker** | `cross-encoder/ms-marco-TinyBERT-L2-v2` |
| **Tokenization** | NLTK Porter Stemmer + custom stopword list |
| **Frontend** | Streamlit |
| **Package Manager** | uv |

---

*Happy searching!* 🍿
