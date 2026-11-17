"""
Answer generation module using Google Gemini API.

Responsibilities:
- Build grounded RAG prompts containing user questions and retrieved contexts.
- Interact with Google Gemini via the official google-genai SDK.
- Configure low temperature to minimize hallucinations.
- Extract and aggregate reference page numbers.
- Handle cases where no relevant context was found without unnecessary API calls.
"""

import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()


NO_CONTEXT_MESSAGE = (
    "O documento não contém informações suficientes para responder a esta pergunta."
)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")


def build_rag_prompt(query: str, context_chunks: list[dict[str, Any]]) -> str:
    """
    Format a grounded prompt containing the retrieved context chunks and the user's query.

    Args:
        query: The user's question.
        context_chunks: List of retrieved chunks with 'text' and 'page_numbers'.

    Returns:
        Structured prompt string instructing the model to ground its answer.
    """
    context_sections: list[str] = []

    for i, chunk in enumerate(context_chunks, start=1):
        pages_str = ", ".join(str(p) for p in chunk.get("page_numbers", []))
        page_header = f"[Trecho {i} - Página(s): {pages_str}]"
        context_sections.append(f"{page_header}\n{chunk['text']}")

    context_text = "\n\n".join(context_sections)

    prompt = f"""Você é um assistente técnico especialista e preciso.
Sua tarefa é responder à pergunta do usuário utilizando EXCLUSIVAMENTE as informações contidas no contexto abaixo.

Diretrizes obrigatórias:
1. Responda de forma clara, objetiva e baseada estritamente no contexto fornecido.
2. Se a resposta NÃO estiver presente no contexto fornecido, responda exatamente: "{NO_CONTEXT_MESSAGE}"
3. Não use conhecimentos prévios ou externos que não estejam explicitamente citados no contexto.
4. Mantenha o idioma da resposta igual ao idioma da pergunta (português por padrão).

--- CONTEXTO DO DOCUMENTO ---
{context_text}

--- PERGUNTA DO USUÁRIO ---
{query}

--- RESPOSTA ---
"""
    return prompt.strip()


def generate_answer(
    query: str,
    context_chunks: list[dict[str, Any]],
    client: genai.Client | None = None,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Generate a grounded answer for a question using the retrieved context and Gemini API.

    Args:
        query: The user's question.
        context_chunks: List of relevant chunks returned by the retriever.
        client: Optional pre-configured genai.Client instance (useful for dependency injection/testing).
        model_name: Gemini model identifier to use for generation.

    Returns:
        Dict containing:
        {
            "answer": str,
            "pages_used": list[int],
            "has_context": bool
        }
    """
    # 1. Short-circuit: if no context was retrieved, avoid unnecessary API calls
    if not context_chunks:
        return {
            "answer": NO_CONTEXT_MESSAGE,
            "pages_used": [],
            "has_context": False,
        }

    # 2. Extract and deduplicate all pages referenced in the context chunks
    pages_set: set[int] = set()
    for chunk in context_chunks:
        for page in chunk.get("page_numbers", []):
            pages_set.add(page)
    pages_used = sorted(list(pages_set))

    # 3. Instantiate client if not injected
    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please configure it in your .env file."
            )
        client = genai.Client(api_key=api_key)

    # 4. Build grounded prompt
    prompt = build_rag_prompt(query, context_chunks)

    # 5. Call Gemini API with low temperature to minimize hallucinations
    config = types.GenerateContentConfig(
        temperature=0.2,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )

    answer_text = response.text.strip() if response.text else NO_CONTEXT_MESSAGE

    return {
        "answer": answer_text,
        "pages_used": pages_used,
        "has_context": True,
    }
