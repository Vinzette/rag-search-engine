import sys
from pathlib import Path

# Allow importing from cli/lib
sys.path.insert(0, str(Path(__file__).resolve().parent / "cli"))

import streamlit as st
import json
import os
import tempfile
import mimetypes
import numpy as np

from lib.search_utils import load_movies, PROJECT_ROOT, BM25_K1, BM25_B
from lib.keyword_search import InvertedIndex, search_command, bm25_search
from lib.semantic_search import (
    SemanticSearch,
    ChunkedSemanticSearch,
    fixed_sized_chunking,
    semantic_chunking,
)
from lib.hybrid_search import HybridSearch, normalize_scores
from lib.llm import (
    answer_question,
    summarize_documents,
    citation_documents,
    detailed_question_answering,
    augment_prompt,
    llm_judge,
)
from lib.rerank import individual_rerank, batch_rerank, cross_encoder_rerank
from lib.multimodal_search import MultimodalSearch

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Search Engine",
    page_icon="🔎",
    layout="wide",
)

# ── Cached resources (heavy objects loaded once) ─────────────────────────────


@st.cache_resource(show_spinner="Loading movies…")
def get_movies():
    return load_movies()


@st.cache_resource(show_spinner="Initializing hybrid search (BM25 + semantic)…")
def get_hybrid_search():
    movies = get_movies()
    return HybridSearch(movies)


@st.cache_resource(show_spinner="Loading BM25 index…")
def get_bm25_index():
    idx = InvertedIndex()
    if not os.path.exists(idx.index_path):
        idx.build()
        idx.save()
    idx.load()
    return idx


@st.cache_resource(show_spinner="Loading semantic search (full-doc)…")
def get_semantic_search():
    movies = get_movies()
    ss = SemanticSearch()
    ss.load_or_create_embeddings(movies)
    return ss


@st.cache_resource(show_spinner="Loading semantic search (chunked)…")
def get_chunked_semantic_search():
    movies = get_movies()
    css = ChunkedSemanticSearch()
    css.load_or_create_chunk_embeddings(movies)
    return css


@st.cache_resource(show_spinner="Loading multimodal (CLIP) model…")
def get_multimodal_search():
    movies = get_movies()
    return MultimodalSearch(movies)


# ── Helpers ──────────────────────────────────────────────────────────────────


def display_result_card(idx, result, show_scores=True):
    """Render a single search result as an expander."""
    title = result.get("title", "Untitled")
    desc = result.get("description", result.get("document", ""))
    with st.expander(f"**{idx}. {title}**", expanded=(idx <= 3)):
        st.write(desc[:500] + ("…" if len(desc) > 500 else ""))
        if show_scores:
            score_items = {
                "RRF": result.get("rrf_score"),
                "BM25 rank": result.get("bm25_rank"),
                "Semantic rank": result.get("sem_rank"),
                "Hybrid": result.get("hybrid_score"),
                "Score": result.get("score"),
                "Cross-enc.": result.get("cross_encoder_score"),
            }
            active = {k: v for k, v in score_items.items() if v is not None}
            if active:
                cols = st.columns(min(len(active), 4))
                for i, (label, val) in enumerate(active.items()):
                    cols[i % len(cols)].metric(
                        label, f"{val:.4f}" if isinstance(val, float) else val
                    )


def describe_image_query(image_bytes, image_mime, query_text):
    """Use Gemini to rewrite a query incorporating an image (describe_image_cli)."""
    from google.genai import types
    from lib.llm import client, model

    system_prompt = (
        "Given the included image and text query, rewrite the text query to improve "
        "search results from a movie database. Make sure to:\n"
        "- Synthesize visual and textual information\n"
        "- Focus on movie-specific details (actors, scenes, style, etc.)\n"
        "- Return only the rewritten query, without any additional commentary"
    )
    parts = [
        system_prompt,
        types.Part.from_bytes(data=image_bytes, mime_type=image_mime),
        query_text.strip(),
    ]
    response = client.models.generate_content(model=model, contents=parts)
    token_count = (
        response.usage_metadata.total_token_count
        if response.usage_metadata
        else None
    )
    return response.text.strip(), token_count


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🔎 RAG Search Engine")
st.sidebar.caption("RAG pipeline — frontend")

page = st.sidebar.radio(
    "Navigate",
    [
        "Keyword Search",
        "Semantic Search",
        "Hybrid Search",
        "RAG (Q&A)",
        "Image / Multimodal",
        "Evaluation",
    ],
    index=0,
)

# ════════════════════════════════════════════════════════════════════════════
#  1. KEYWORD SEARCH  (keyword_search_cli.py)
# ════════════════════════════════════════════════════════════════════════════

if page == "Keyword Search":
    st.header("🔤 Keyword Search")

    command = st.selectbox(
        "Command",
        ["search", "bm25search", "build", "tf", "idf", "tfidf", "bm25tf", "bm25idf"],
    )

    # ── search ───────────────────────────────────────────────────────────
    if command == "search":
        st.subheader("Simple keyword search")
        query = st.text_input("Query")
        limit = st.slider("Limit", 1, 20, 5)
        if st.button("Search", type="primary", disabled=not query):
            with st.spinner("Searching…"):
                results = search_command(query, limit)
            if results:
                for i, r in enumerate(results, 1):
                    st.write(f"**{i}. {r['title']}**")
            else:
                st.warning("No results found.")

    # ── bm25search ───────────────────────────────────────────────────────
    elif command == "bm25search":
        st.subheader("BM25 search")
        query = st.text_input("Query")
        limit = st.slider("Limit", 1, 20, 5)
        if st.button("Search", type="primary", disabled=not query):
            with st.spinner("Searching…"):
                results = bm25_search(query, limit)
            if results:
                for i, r in enumerate(results, 1):
                    display_result_card(i, r)
            else:
                st.warning("No results found.")

    # ── build ────────────────────────────────────────────────────────────
    elif command == "build":
        st.subheader("Build inverted index")
        st.caption("Builds the BM25 inverted index from movies.json and saves pickle files to `cache/`.")
        if st.button("Build Index", type="primary"):
            with st.spinner("Building index…"):
                idx = InvertedIndex()
                idx.build()
                idx.save()
            st.success("Index built and saved to cache/")

    # ── tf ───────────────────────────────────────────────────────────────
    elif command == "tf":
        st.subheader("Term Frequency (TF)")
        col1, col2 = st.columns(2)
        doc_id = col1.number_input("Document ID", min_value=0, step=1, value=0)
        term = col2.text_input("Term")
        if st.button("Calculate", type="primary", disabled=not term):
            with st.spinner("Loading index…"):
                idx = get_bm25_index()
            tf_val = idx.get_tf(doc_id, term)
            st.metric("TF", tf_val)

    # ── idf ──────────────────────────────────────────────────────────────
    elif command == "idf":
        st.subheader("Inverse Document Frequency (IDF)")
        term = st.text_input("Term")
        if st.button("Calculate", type="primary", disabled=not term):
            with st.spinner("Loading index…"):
                idx = get_bm25_index()
            idf_val = idx.get_idf(term)
            st.metric("IDF", f"{idf_val:.4f}")

    # ── tfidf ────────────────────────────────────────────────────────────
    elif command == "tfidf":
        st.subheader("TF-IDF")
        col1, col2 = st.columns(2)
        doc_id = col1.number_input("Document ID", min_value=0, step=1, value=0)
        term = col2.text_input("Term")
        if st.button("Calculate", type="primary", disabled=not term):
            with st.spinner("Loading index…"):
                idx = get_bm25_index()
            tfidf_val = idx.get_tfidf(doc_id, term)
            st.metric("TF-IDF", f"{tfidf_val:.4f}")

    # ── bm25tf ───────────────────────────────────────────────────────────
    elif command == "bm25tf":
        st.subheader("BM25 Term Frequency")
        col1, col2 = st.columns(2)
        doc_id = col1.number_input("Document ID", min_value=0, step=1, value=0)
        term = col2.text_input("Term")
        col3, col4 = st.columns(2)
        k1 = col3.number_input("k1", value=BM25_K1, format="%.2f")
        b = col4.number_input("b", value=BM25_B, format="%.2f")
        if st.button("Calculate", type="primary", disabled=not term):
            with st.spinner("Loading index…"):
                idx = get_bm25_index()
            bm25_tf_val = idx.get_bm25_tf(doc_id, term, k1, b)
            st.metric("BM25 TF", f"{bm25_tf_val:.4f}")

    # ── bm25idf ──────────────────────────────────────────────────────────
    elif command == "bm25idf":
        st.subheader("BM25 Inverse Document Frequency")
        term = st.text_input("Term")
        if st.button("Calculate", type="primary", disabled=not term):
            with st.spinner("Loading index…"):
                idx = get_bm25_index()
            bm25_idf_val = idx.get_bm25_idf(term)
            st.metric("BM25 IDF", f"{bm25_idf_val:.4f}")


# ════════════════════════════════════════════════════════════════════════════
#  2. SEMANTIC SEARCH  (semantic_search_cli.py)
# ════════════════════════════════════════════════════════════════════════════

elif page == "Semantic Search":
    st.header("🧠 Semantic Search")

    command = st.selectbox(
        "Command",
        [
            "search",
            "search_chunked",
            "embed_text",
            "embedquery",
            "chunk",
            "semantic_chunk",
            "embed_chunks",
            "verify",
            "verify_embeddings",
        ],
    )

    # ── search (full-doc) ────────────────────────────────────────────────
    if command == "search":
        st.subheader("Semantic search (full-document embeddings)")
        query = st.text_input("Query")
        limit = st.slider("Limit", 1, 20, 5)
        if st.button("Search", type="primary", disabled=not query):
            with st.spinner("Searching…"):
                ss = get_semantic_search()
                results = ss.search(query, limit)
            if results:
                for i, r in enumerate(results, 1):
                    display_result_card(i, r)
            else:
                st.warning("No results found.")

    # ── search_chunked ───────────────────────────────────────────────────
    elif command == "search_chunked":
        st.subheader("Semantic search (chunked embeddings)")
        query = st.text_input("Query")
        limit = st.slider("Limit", 1, 20, 5)
        if st.button("Search", type="primary", disabled=not query):
            with st.spinner("Searching…"):
                css = get_chunked_semantic_search()
                raw = css.search_chunks(query, limit)
                results = [
                    {
                        "title": r["title"],
                        "description": r.get("document", ""),
                        "score": r["score"],
                    }
                    for r in raw
                ]
            if results:
                for i, r in enumerate(results, 1):
                    display_result_card(i, r)
            else:
                st.warning("No results found.")

    # ── embed_text ───────────────────────────────────────────────────────
    elif command == "embed_text":
        st.subheader("Encode text → embedding vector")
        text = st.text_area("Text to encode")
        if st.button("Embed", type="primary", disabled=not text):
            with st.spinner("Encoding…"):
                ss = get_semantic_search()
                embedding = ss.generate_embedding(text)
            st.write(f"**Dimensions:** {embedding.shape[0]}")
            st.write(f"**First 3 values:** `{embedding[:3]}`")
            with st.expander("Full vector"):
                st.write(embedding.tolist())

    # ── embedquery ───────────────────────────────────────────────────────
    elif command == "embedquery":
        st.subheader("Encode query → embedding vector")
        query = st.text_input("Query to encode")
        if st.button("Embed", type="primary", disabled=not query):
            with st.spinner("Encoding…"):
                ss = get_semantic_search()
                embedding = ss.generate_embedding(query)
            st.write(f"**Shape:** {embedding.shape}")
            st.write(f"**First 5 dimensions:** `{embedding[:5]}`")

    # ── chunk (fixed-size) ───────────────────────────────────────────────
    elif command == "chunk":
        st.subheader("Fixed-size text chunking")
        text = st.text_area("Text to chunk")
        col1, col2 = st.columns(2)
        chunk_size = col1.number_input("Chunk size (words)", value=200, min_value=1)
        overlap = col2.number_input("Overlap (words)", value=0, min_value=0)
        if st.button("Chunk", type="primary", disabled=not text):
            chunks = fixed_sized_chunking(text, overlap, chunk_size)
            st.write(f"**{len(chunks)} chunk(s)** from {len(text)} characters")
            for i, chunk in enumerate(chunks, 1):
                st.info(f"**Chunk {i}:** {chunk}")

    # ── semantic_chunk ───────────────────────────────────────────────────
    elif command == "semantic_chunk":
        st.subheader("Semantic text chunking (sentence-based)")
        text = st.text_area("Text to chunk")
        col1, col2 = st.columns(2)
        max_chunk_size = col1.number_input("Max chunk size (sentences)", value=4, min_value=1)
        overlap = col2.number_input("Overlap (sentences)", value=0, min_value=0)
        if st.button("Chunk", type="primary", disabled=not text):
            chunks = semantic_chunking(text, max_chunk_size, overlap)
            st.write(f"**{len(chunks)} chunk(s)** from {len(text)} characters")
            for i, chunk in enumerate(chunks, 1):
                st.info(f"**Chunk {i}:** {chunk}")

    # ── embed_chunks ─────────────────────────────────────────────────────
    elif command == "embed_chunks":
        st.subheader("Build / load chunk embeddings")
        st.caption("Creates semantic chunk embeddings for all movies and saves to cache/.")
        if st.button("Embed Chunks", type="primary"):
            with st.spinner("Building chunk embeddings…"):
                movies = get_movies()
                css = ChunkedSemanticSearch()
                embeddings = css.load_or_create_chunk_embeddings(movies)
            st.success(f"Generated **{len(embeddings)}** chunked embeddings.")

    # ── verify ───────────────────────────────────────────────────────────
    elif command == "verify":
        st.subheader("Verify embedding model")
        if st.button("Verify", type="primary"):
            with st.spinner("Loading model…"):
                ss = get_semantic_search()
            st.success("Model loaded successfully!")
            st.write(f"**Model:** `{ss.model}`")
            st.write(f"**Max sequence length:** {ss.model.max_seq_length}")

    # ── verify_embeddings ────────────────────────────────────────────────
    elif command == "verify_embeddings":
        st.subheader("Verify saved embeddings")
        if st.button("Verify", type="primary"):
            with st.spinner("Loading…"):
                ss = get_semantic_search()
                movies = get_movies()
            st.write(f"**Number of docs:** {len(movies)}")
            st.write(
                f"**Embeddings shape:** {ss.embeddings.shape[0]} vectors × {ss.embeddings.shape[1]} dimensions"
            )


# ════════════════════════════════════════════════════════════════════════════
#  3. HYBRID SEARCH  (hybrid_search_cli.py)
# ════════════════════════════════════════════════════════════════════════════

elif page == "Hybrid Search":
    st.header("⚡ Hybrid Search")

    command = st.selectbox(
        "Command", ["rrf-search", "weighted-search", "normalize"]
    )

    # ── rrf-search ───────────────────────────────────────────────────────
    if command == "rrf-search":
        st.subheader("RRF Hybrid Search")
        query = st.text_input("Query")
        col1, col2 = st.columns(2)
        k_param = col1.number_input("RRF k parameter", value=60, min_value=1)
        limit = col2.slider("Results to return", 1, 20, 5)

        col3, col4 = st.columns(2)
        enhance = col3.selectbox("Query enhancement", [None, "spell", "rewrite", "expand"])
        rerank_method = col4.selectbox(
            "Re-rank method", [None, "individual", "batch", "cross_encoder"]
        )

        debug = st.text_input("Debug — track a title (optional)", value="")
        evaluate = st.checkbox("Run LLM-as-judge evaluation on results")

        if st.button("Search", type="primary", disabled=not query):
            hs = get_hybrid_search()
            actual_query = query

            # Enhancement
            if enhance:
                with st.spinner(f"Enhancing query ({enhance})…"):
                    actual_query = augment_prompt(query, enhance)
                st.info(f"Enhanced query: **{actual_query}**")

            # Search
            rrf_limit = limit * 5 if rerank_method else limit
            with st.spinner("Running RRF search…"):
                results = hs.rrf_search(actual_query, k_param, rrf_limit)

            # Debug tracking
            if debug:
                found_pos = None
                for didx, r in enumerate(results):
                    if debug.lower().strip() in r["title"].lower().strip():
                        found_pos = didx
                        break
                if found_pos is not None:
                    st.caption(f"🐛 DEBUG: **{debug}** found at position **{found_pos}** after hybrid search")
                else:
                    st.caption(f"🐛 DEBUG: **{debug}** not found in hybrid results")

            # Reranking
            if rerank_method:
                with st.spinner(f"Re-ranking with **{rerank_method}**…"):
                    if rerank_method == "individual":
                        results = individual_rerank(actual_query, results)
                    elif rerank_method == "batch":
                        results = batch_rerank(actual_query, results)
                    elif rerank_method == "cross_encoder":
                        results = cross_encoder_rerank(actual_query, results)

                if debug:
                    found_pos = None
                    for didx, r in enumerate(results):
                        if debug.lower().strip() in r["title"].lower().strip():
                            found_pos = didx
                            break
                    if found_pos is not None:
                        st.caption(f"🐛 DEBUG: **{debug}** found at position **{found_pos}** after reranking")
                    else:
                        st.caption(f"🐛 DEBUG: **{debug}** not found after reranking")

            results = results[:limit]

            # Display results
            if results:
                st.subheader(f"Top {len(results)} results")
                for i, r in enumerate(results, 1):
                    display_result_card(i, r)

                # LLM-as-judge evaluation
                if evaluate:
                    with st.spinner("Running LLM-as-judge evaluation…"):
                        formatted_results = [
                            f"<result id={i}>{r['title']}: {r.get('description', '')[:100]}</result>"
                            for i, r in enumerate(results, 1)
                        ]
                        llm_results = llm_judge(actual_query, "\n".join(formatted_results))

                    st.subheader("LLM Judge Scores")
                    for i, r in enumerate(results, 1):
                        score = llm_results[i - 1] if i - 1 < len(llm_results) else "?"
                        st.write(f"**{i}. {r['title']}:** {score}/3")
            else:
                st.warning("No results found.")

    # ── weighted-search ──────────────────────────────────────────────────
    elif command == "weighted-search":
        st.subheader("Weighted Hybrid Search")
        query = st.text_input("Query")
        col1, col2 = st.columns(2)
        alpha = col1.slider("Alpha (BM25 weight)", 0.0, 1.0, 0.5, 0.05)
        limit = col2.slider("Results to return", 1, 20, 5)

        if st.button("Search", type="primary", disabled=not query):
            with st.spinner("Searching…"):
                hs = get_hybrid_search()
                results = hs.weighted_search(query, alpha, limit)
            if results:
                st.subheader(f"Top {len(results[:limit])} results")
                for i, r in enumerate(results[:limit], 1):
                    display_result_card(i, r)
            else:
                st.warning("No results found.")

    # ── normalize ────────────────────────────────────────────────────────
    elif command == "normalize":
        st.subheader("Normalize a list of scores")
        scores_str = st.text_input("Scores (space-separated)", placeholder="0.5 1.2 3.7 0.1")
        if st.button("Normalize", type="primary", disabled=not scores_str):
            try:
                scores = [float(s) for s in scores_str.strip().split()]
            except ValueError:
                st.error("Please enter valid numbers separated by spaces.")
                scores = None
            if scores:
                norm = normalize_scores(scores)
                col_orig, col_norm = st.columns(2)
                col_orig.write("**Original**")
                col_norm.write("**Normalized**")
                for orig, n in zip(scores, norm):
                    col_orig.write(f"{orig}")
                    col_norm.write(f"{n:.4f}")


# ════════════════════════════════════════════════════════════════════════════
#  4. RAG (Q&A)  (augmented_generation_cli.py)
# ════════════════════════════════════════════════════════════════════════════

elif page == "RAG (Q&A)":
    st.header("🤖 Retrieval-Augmented Generation")

    command = st.selectbox(
        "Command", ["rag", "summarize", "citations", "question"]
    )

    query = st.text_input("Enter your question / query")
    limit = st.slider("Number of documents to retrieve", 1, 15, 5)

    if st.button("Generate", type="primary", disabled=not query):
        hs = get_hybrid_search()

        with st.spinner("Retrieving relevant documents…"):
            rrf_results = hs.rrf_search(query, k=60, limit=limit)

        with st.expander("Retrieved documents", expanded=False):
            for r in rrf_results:
                st.write(f"- **{r['title']}**")

        with st.spinner("Generating response with Gemini…"):
            if command == "rag":
                response = answer_question(query, rrf_results)
            elif command == "summarize":
                response = summarize_documents(query, rrf_results)
            elif command == "citations":
                response = citation_documents(query, rrf_results)
            elif command == "question":
                response = detailed_question_answering(query, rrf_results)

        st.subheader("Response")
        st.markdown(response)


# ════════════════════════════════════════════════════════════════════════════
#  5. IMAGE / MULTIMODAL  (multimodal_search_cli.py + describe_image_cli.py)
# ════════════════════════════════════════════════════════════════════════════

elif page == "Image / Multimodal":
    st.header("🖼️ Image & Multimodal")

    command = st.selectbox(
        "Command", ["image_search", "describe_image"]
    )

    # ── image_search ─────────────────────────────────────────────────────
    if command == "image_search":
        st.subheader("Image search (CLIP)")
        st.caption("Upload an image and find the most similar movies using CLIP embeddings.")
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
        limit = st.slider("Results to return", 1, 20, 5)

        if uploaded is not None:
            st.image(uploaded, caption="Uploaded image", width=300)

            if st.button("Search by image", type="primary"):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                with st.spinner("Encoding image & searching…"):
                    ms = get_multimodal_search()
                    results = ms.search_with_image(tmp_path, limit)

                if results:
                    st.subheader(f"Top {len(results)} matches")
                    for i, r in enumerate(results, 1):
                        display_result_card(i, r)
                else:
                    st.warning("No results found.")

    # ── describe_image (query rewriting with Gemini) ─────────────────────
    elif command == "describe_image":
        st.subheader("Image + Text → Rewritten Query (Gemini)")
        st.caption(
            "Upload an image and provide a text query. Gemini rewrites the query "
            "to improve movie database search by incorporating visual context."
        )
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
        query = st.text_input("Text query to go with the image")

        if uploaded is not None:
            st.image(uploaded, caption="Uploaded image", width=300)

        if st.button("Rewrite Query", type="primary", disabled=(not uploaded or not query)):
            mime, _ = mimetypes.guess_type(uploaded.name)
            mime = mime or "image/jpeg"
            img_bytes = uploaded.getvalue()

            with st.spinner("Generating rewritten query with Gemini…"):
                rewritten, token_count = describe_image_query(img_bytes, mime, query)

            st.success(f"**Rewritten query:** {rewritten}")
            if token_count is not None:
                st.caption(f"Total tokens: {token_count}")


# ════════════════════════════════════════════════════════════════════════════
#  6. EVALUATION  (evaluation_cli.py)
# ════════════════════════════════════════════════════════════════════════════

elif page == "Evaluation":
    st.header("📊 Search Evaluation")
    st.caption("Runs precision@k, recall@k, and F1 against the golden dataset.")

    limit = st.slider("k (for precision@k / recall@k)", 1, 20, 5)

    if st.button("Run evaluation", type="primary"):
        with open(PROJECT_ROOT / "data" / "golden_dataset.json") as f:
            test_cases = json.load(f)["test_cases"]

        hs = get_hybrid_search()

        progress = st.progress(0, text="Evaluating…")
        rows = []
        for i, tc in enumerate(test_cases):
            qry = tc["query"]
            expected = tc["relevant_docs"]
            rrf_results = hs.rrf_search(qry, k=60, limit=limit)
            retrieved_titles = [r["title"] for r in rrf_results]
            relevant_cnt = sum(1 for t in retrieved_titles if t in expected)
            precision = relevant_cnt / limit
            recall = relevant_cnt / len(expected) if expected else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            rows.append(
                {
                    "Query": qry,
                    f"Precision@{limit}": round(precision, 4),
                    f"Recall@{limit}": round(recall, 4),
                    "F1": round(f1, 4),
                    "Retrieved": ", ".join(retrieved_titles),
                    "Expected": ", ".join(expected),
                }
            )
            progress.progress(
                (i + 1) / len(test_cases), text=f"Evaluated {i + 1}/{len(test_cases)}"
            )

        progress.empty()

        st.subheader("Results")
        st.dataframe(rows, use_container_width=True)

        # Averages
        avg_p = sum(r[f"Precision@{limit}"] for r in rows) / len(rows)
        avg_r = sum(r[f"Recall@{limit}"] for r in rows) / len(rows)
        avg_f = sum(r["F1"] for r in rows) / len(rows)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(f"Avg Precision@{limit}", f"{avg_p:.4f}")
        mc2.metric(f"Avg Recall@{limit}", f"{avg_r:.4f}")
        mc3.metric("Avg F1", f"{avg_f:.4f}")
