"""
Text chunking module.

Responsibilities:
- Take structured pages extracted from a PDF.
- Split the continuous text into overlapping chunks.
- Prevent breaking words in the middle (snap to whitespace/newline).
- Map each chunk back to the source page number(s) it spans.
"""

from typing import Any


def split_text_with_overlap(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[tuple[int, int, str]]:
    """
    Split text into chunks with character overlap, avoiding word slicing.

    Args:
        text: Input string to be chunked.
        chunk_size: Maximum desired character length per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of tuples: (start_index, end_index, chunk_text)

    Raises:
        ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})"
        )

    text_length = len(text)
    if not text.strip():
        return []

    chunks: list[tuple[int, int, str]] = []
    start = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # If we are not at the very end of the text, try to snap to nearest whitespace
        if end < text_length:
            # First try paragraph/line break, then regular space
            snap = text.rfind("\n", start, end)
            if snap == -1 or snap <= start:
                snap = text.rfind(" ", start, end)

            # Snap only if a valid delimiter was found within this window
            if snap > start:
                end = snap

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append((start, end, chunk_text))

        if end >= text_length:
            break

        # Move forward, leaving the overlap region for the next chunk
        new_start = max(start + 1, end - chunk_overlap)

        # If new_start falls in the middle of a word, advance to the start of the next word
        if new_start < end and new_start > 0:
            if not text[new_start].isspace() and not text[new_start - 1].isspace():
                # We are inside a word: find the next whitespace to snap cleanly
                next_space = text.find(" ", new_start, end)
                if next_space != -1:
                    # Skip any consecutive whitespace
                    while next_space < end and text[next_space].isspace():
                        next_space += 1
                    new_start = next_space

        start = new_start

    return chunks


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, Any]]:
    """
    Chunk extracted pages into overlapping text pieces while preserving page references.

    Args:
        pages: List of dicts, e.g. [{"page_number": 1, "text": "..."}]
        chunk_size: Maximum character size per chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.

    Returns:
        List of dicts:
        [
            {
                "chunk_id": int,
                "text": str,
                "page_numbers": list[int]
            },
            ...
        ]
    """
    if not pages:
        return []

    # 1. Build a continuous text stream while recording character boundaries for each page
    full_text_parts: list[str] = []
    page_spans: list[tuple[int, int, int]] = []  # (page_number, start_pos, end_pos)
    current_pos = 0

    for page in pages:
        page_num = page["page_number"]
        page_text = page["text"]

        start_pos = current_pos
        full_text_parts.append(page_text)
        current_pos += len(page_text)
        end_pos = current_pos

        page_spans.append((page_num, start_pos, end_pos))

        # Separate pages with double newline in the continuous stream
        full_text_parts.append("\n\n")
        current_pos += 2

    full_text = "".join(full_text_parts)

    # 2. Split continuous text into overlapping chunks
    raw_chunks = split_text_with_overlap(
        full_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 3. Associate each chunk with the page(s) it intersects with
    structured_chunks: list[dict[str, Any]] = []

    for chunk_id, (chunk_start, chunk_end, chunk_text) in enumerate(raw_chunks):
        matching_pages: list[int] = []

        for page_num, p_start, p_end in page_spans:
            # Overlap condition: chunk_start < p_end AND chunk_end > p_start
            if max(chunk_start, p_start) < min(chunk_end, p_end):
                matching_pages.append(page_num)

        structured_chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "page_numbers": sorted(list(set(matching_pages))),
        })

    return structured_chunks
