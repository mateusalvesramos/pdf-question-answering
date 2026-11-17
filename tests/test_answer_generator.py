"""
Unit tests for the answer_generator module.

Test strategy:
- Test prompt construction and grounding instructions.
- Test short-circuit behavior when no chunks are provided (no API call made).
- Use unittest.mock to mock the Gemini Client and API responses (deterministic, no API key needed).
- Verify page aggregation and deduplication.
- Verify missing API key handling.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.answer_generator import (
    NO_CONTEXT_MESSAGE,
    build_rag_prompt,
    generate_answer,
)


@pytest.fixture
def sample_context_chunks():
    return [
        {
            "chunk_id": 0,
            "text": "Supervised learning algorithms include Linear Regression and Decision Trees.",
            "page_numbers": [1, 2],
        },
        {
            "chunk_id": 1,
            "text": "Neural networks are composed of layers of nodes inspired by biological neurons.",
            "page_numbers": [2, 3],
        },
    ]


# ---------------------------------------------------------------------------
# Prompt Construction Tests
# ---------------------------------------------------------------------------

class TestBuildRagPrompt:

    def test_prompt_contains_query_and_chunks(self, sample_context_chunks):
        query = "Quais sao os algoritmos citados?"
        prompt = build_rag_prompt(query, sample_context_chunks)

        assert query in prompt
        assert "Linear Regression" in prompt
        assert "Neural networks" in prompt
        assert "Página(s): 1, 2" in prompt
        assert "Página(s): 2, 3" in prompt

    def test_prompt_contains_grounding_directives(self, sample_context_chunks):
        prompt = build_rag_prompt("teste", sample_context_chunks)
        # Check that grounding rules and fallback instructions are in the prompt
        assert "EXCLUSIVAMENTE" in prompt
        assert NO_CONTEXT_MESSAGE in prompt


# ---------------------------------------------------------------------------
# Answer Generation Tests (Mocked API)
# ---------------------------------------------------------------------------

class TestGenerateAnswer:

    def test_empty_chunks_returns_no_context_without_calling_api(self):
        # Pass a mock client to ensure it is NEVER called when chunks are empty
        mock_client = MagicMock()

        result = generate_answer("qualquer pergunta", [], client=mock_client)

        assert result["has_context"] is False
        assert result["pages_used"] == []
        assert result["answer"] == NO_CONTEXT_MESSAGE
        # Ensure generate_content was not invoked
        mock_client.models.generate_content.assert_not_called()

    def test_successful_generation_with_mocked_gemini(self, sample_context_chunks):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Os algoritmos citados sao Linear Regression e Decision Trees."
        mock_client.models.generate_content.return_value = mock_response

        result = generate_answer(
            query="Quais algoritmos sao citados?",
            context_chunks=sample_context_chunks,
            client=mock_client,
        )

        assert result["has_context"] is True
        assert result["answer"] == "Os algoritmos citados sao Linear Regression e Decision Trees."
        # Pages [1, 2] from chunk 0 and [2, 3] from chunk 1 must be deduplicated and sorted -> [1, 2, 3]
        assert result["pages_used"] == [1, 2, 3]

        # Verify call arguments passed to Gemini
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.5-flash"
        assert "Linear Regression" in call_kwargs["contents"]
        assert call_kwargs["config"].temperature == 0.2

    def test_missing_api_key_raises_value_error(self, sample_context_chunks):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GEMINI_API_KEY not found"):
                generate_answer(
                    query="Pergunta",
                    context_chunks=sample_context_chunks,
                    client=None,
                )

    def test_empty_response_text_falls_back_to_no_context_message(self, sample_context_chunks):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""  # Model returned empty string
        mock_client.models.generate_content.return_value = mock_response

        result = generate_answer(
            query="Alguma pergunta",
            context_chunks=sample_context_chunks,
            client=mock_client,
        )

        assert result["answer"] == NO_CONTEXT_MESSAGE
