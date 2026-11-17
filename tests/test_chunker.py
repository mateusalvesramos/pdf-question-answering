"""
Unit tests for the chunker module.

Test strategy:
- Validate parameter boundary conditions and error handling.
- Verify sliding window overlap behavior.
- Ensure words are not chopped in half at chunk boundaries.
- Verify page tracking across single-page and cross-page chunks.
"""

import pytest

from src.chunker import chunk_pages, split_text_with_overlap


# ---------------------------------------------------------------------------
# Tests for split_text_with_overlap
# ---------------------------------------------------------------------------

class TestSplitTextWithOverlap:

    def test_invalid_parameters_raise_value_error(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            split_text_with_overlap("sample text", chunk_size=0)

        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            split_text_with_overlap("sample text", chunk_size=100, chunk_overlap=-1)

        with pytest.raises(ValueError, match="chunk_overlap.*must be strictly smaller"):
            split_text_with_overlap("sample text", chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap.*must be strictly smaller"):
            split_text_with_overlap("sample text", chunk_size=100, chunk_overlap=150)

    def test_empty_or_whitespace_returns_empty_list(self):
        assert split_text_with_overlap("") == []
        assert split_text_with_overlap("   \n\t  ") == []

    def test_text_smaller_than_chunk_size_returns_single_chunk(self):
        text = "Short sentence that easily fits in one chunk."
        chunks = split_text_with_overlap(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) == 1
        assert chunks[0][2] == text

    def test_overlap_contains_repeated_context(self):
        # 10 words, ~50 chars total
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        # Force a small chunk size to generate multiple chunks
        chunks = split_text_with_overlap(text, chunk_size=30, chunk_overlap=10)

        assert len(chunks) > 1

        # Check that consecutive chunks have overlapping content
        for i in range(len(chunks) - 1):
            current_chunk_text = chunks[i][2]
            next_chunk_text = chunks[i + 1][2]

            # The start of next chunk or end of current chunk must share words
            words_in_current = set(current_chunk_text.split())
            words_in_next = set(next_chunk_text.split())
            shared_words = words_in_current.intersection(words_in_next)

            assert len(shared_words) > 0, "Consecutive chunks must share words due to overlap"

    def test_does_not_break_words_in_middle_when_spaces_exist(self):
        text = "internationalization and compartmentalization are very long words"
        chunks = split_text_with_overlap(text, chunk_size=30, chunk_overlap=5)

        for _, _, chunk_text in chunks:
            # None of the words should end in a partial word from the source sentence
            words = chunk_text.split()
            for word in words:
                assert word in text.split(), f"Word '{word}' was sliced in half!"


# ---------------------------------------------------------------------------
# Tests for chunk_pages
# ---------------------------------------------------------------------------

class TestChunkPages:

    def test_empty_pages_returns_empty(self):
        assert chunk_pages([]) == []

    def test_single_page_single_chunk(self):
        pages = [{"page_number": 1, "text": "This is page one."}]
        result = chunk_pages(pages, chunk_size=500, chunk_overlap=50)

        assert len(result) == 1
        assert result[0]["chunk_id"] == 0
        assert result[0]["text"] == "This is page one."
        assert result[0]["page_numbers"] == [1]

    def test_chunk_spanning_multiple_pages(self):
        """
        When two pages each have small text, but together they span across a chunk boundary,
        the chunk that includes text from both should have [1, 2] in page_numbers.
        """
        pages = [
            {"page_number": 1, "text": "Page one ends here with some text."},
            {"page_number": 2, "text": "Page two starts here with other text."},
        ]

        # Use a chunk size large enough to cover text from both pages
        result = chunk_pages(pages, chunk_size=1000, chunk_overlap=50)

        assert len(result) == 1
        assert result[0]["page_numbers"] == [1, 2]
        assert "Page one" in result[0]["text"]
        assert "Page two" in result[0]["text"]

    def test_chunk_ids_are_sequential(self):
        pages = [
            {"page_number": 1, "text": "A" * 600},
            {"page_number": 2, "text": "B" * 600},
        ]
        result = chunk_pages(pages, chunk_size=300, chunk_overlap=50)

        assert len(result) > 2
        for i, chunk in enumerate(result):
            assert chunk["chunk_id"] == i
            assert isinstance(chunk["text"], str)
            assert isinstance(chunk["page_numbers"], list)
