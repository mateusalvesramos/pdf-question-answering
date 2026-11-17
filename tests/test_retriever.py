"""
Unit tests for the retriever module.

Test strategy:
- Validate input validation and error raising.
- Test empty queries and empty chunk collections.
- Verify relevance ranking using clearly differentiated thematic documents.
- Verify that irrelevant queries (no shared vocabulary) return no results.
- Verify top_k limits and score filtering.
- Ensure page_numbers metadata is preserved.
"""

import pytest

from src.retriever import retrieve_relevant_chunks


@pytest.fixture
def sample_chunks():
    """Sample chunks on distinct topics to test ranking and retrieval."""
    return [
        {
            "chunk_id": 0,
            "text": "Supervised learning trains models on labeled datasets with inputs and known outputs.",
            "page_numbers": [1],
        },
        {
            "chunk_id": 1,
            "text": "Convolutional neural networks and deep learning excel at image recognition tasks.",
            "page_numbers": [2],
        },
        {
            "chunk_id": 2,
            "text": "To bake homemade sourdough bread, mix flour, water, salt, and active starter yeast.",
            "page_numbers": [3],
        },
    ]


# ---------------------------------------------------------------------------
# Validation & Edge Cases
# ---------------------------------------------------------------------------

class TestRetrieverValidation:

    def test_invalid_top_k_raises_value_error(self, sample_chunks):
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            retrieve_relevant_chunks("machine learning", sample_chunks, top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            retrieve_relevant_chunks("machine learning", sample_chunks, top_k=-2)

    def test_negative_min_score_raises_value_error(self, sample_chunks):
        with pytest.raises(ValueError, match="min_score cannot be negative"):
            retrieve_relevant_chunks("query", sample_chunks, min_score=-0.1)

    def test_empty_query_returns_empty_list(self, sample_chunks):
        assert retrieve_relevant_chunks("", sample_chunks) == []
        assert retrieve_relevant_chunks("   \t\n  ", sample_chunks) == []

    def test_empty_chunks_returns_empty_list(self):
        assert retrieve_relevant_chunks("some question", []) == []


# ---------------------------------------------------------------------------
# Relevance & Ranking Tests
# ---------------------------------------------------------------------------

class TestRetrieverRanking:

    def test_retrieves_most_relevant_chunk_first(self, sample_chunks):
        # Query about neural networks and images
        query = "How do neural networks work for image recognition?"
        results = retrieve_relevant_chunks(query, sample_chunks, top_k=3)

        assert len(results) > 0
        # Chunk 1 (neural networks / image recognition) must be ranked #1
        assert results[0]["chunk_id"] == 1
        assert results[0]["page_numbers"] == [2]
        assert results[0]["score"] > 0.0

    def test_retrieves_baking_chunk_for_recipe_query(self, sample_chunks):
        query = "What ingredients are needed for sourdough bread?"
        results = retrieve_relevant_chunks(query, sample_chunks, top_k=3)

        assert len(results) > 0
        # Chunk 2 (bread recipe) must be ranked #1
        assert results[0]["chunk_id"] == 2
        assert results[0]["page_numbers"] == [3]

    def test_top_k_limits_result_count(self, sample_chunks):
        query = "learning models"
        results = retrieve_relevant_chunks(query, sample_chunks, top_k=1)
        assert len(results) == 1

    def test_query_with_no_vocabulary_overlap_returns_empty(self, sample_chunks):
        """When the query shares no words with the documents, score is 0.0 and filtered out."""
        query = "Astronomy astrophysics black holes in the galaxy"
        results = retrieve_relevant_chunks(query, sample_chunks, top_k=3)
        assert results == []

    def test_scores_are_sorted_descending(self, sample_chunks):
        query = "learning models and neural networks"
        results = retrieve_relevant_chunks(query, sample_chunks, top_k=3)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_min_score_filters_weak_matches(self, sample_chunks):
        query = "supervised learning models"
        # Using a very high threshold that only exact strong matches can pass
        strict_results = retrieve_relevant_chunks(query, sample_chunks, min_score=0.99)
        assert len(strict_results) == 0
