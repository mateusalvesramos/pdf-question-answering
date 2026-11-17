"""
Context retrieval module using TF-IDF and Cosine Similarity.

Responsibilities:
- Transform text chunks and user queries into TF-IDF vector representations.
- Compute cosine similarity between the query and all chunks.
- Rank chunks by relevance score and return top-k matches.
- Filter out completely irrelevant chunks (similarity == 0).
"""

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_relevant_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = 3,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Retrieve the top-k most relevant chunks for a given query using TF-IDF.

    Args:
        query: The user's question or search prompt.
        chunks: List of chunk dictionaries containing at least 'chunk_id', 'text', and 'page_numbers'.
        top_k: Maximum number of relevant chunks to return.
        min_score: Minimum similarity threshold (exclusive) to consider a chunk relevant.

    Returns:
        List of matching chunk dicts, ordered by descending similarity score:
        [
            {
                "chunk_id": int,
                "text": str,
                "page_numbers": list[int],
                "score": float
            },
            ...
        ]

    Raises:
        ValueError: If top_k <= 0 or min_score < 0.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k}")
    if min_score < 0:
        raise ValueError(f"min_score cannot be negative, got {min_score}")

    if not query.strip() or not chunks:
        return []

    # 1. Prepare corpus from chunks
    corpus = [chunk["text"] for chunk in chunks]

    # 2. Fit TF-IDF on all chunks and vectorize both chunks and query
    vectorizer = TfidfVectorizer()
    try:
        chunk_vectors = vectorizer.fit_transform(corpus)
        query_vector = vectorizer.transform([query])
    except ValueError:
        # Happens if corpus has only empty/stopword tokens that scikit-learn discards
        return []

    # 3. Compute cosine similarity between query and all chunks
    # Shape of similarities: (1, num_chunks) -> flatten to 1D array of shape (num_chunks,)
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()

    # 4. Rank indices by descending similarity score
    ranked_indices = similarities.argsort()[::-1]

    # 5. Collect top-k chunks that meet the minimum score threshold
    results: list[dict[str, Any]] = []

    for idx in ranked_indices:
        score = float(similarities[idx])

        # Stop or ignore if score does not exceed the minimum required threshold
        if score <= min_score:
            continue

        results.append({
            "chunk_id": chunks[idx]["chunk_id"],
            "text": chunks[idx]["text"],
            "page_numbers": chunks[idx]["page_numbers"],
            "score": round(score, 4),
        })

        if len(results) >= top_k:
            break

    return results
