"""
PDF text extraction module.

Responsibilities:
- Validate that the given file exists and is a readable PDF.
- Extract plain text from each page using PyMuPDF.
- Return a structured list preserving per-page metadata.
"""

from pathlib import Path

import fitz  # PyMuPDF


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be opened or read."""


class EmptyDocumentError(Exception):
    """Raised when a PDF yields no extractable text at all."""


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """
    Extract text from each page of a PDF file.

    Args:
        pdf_path: Path to the PDF file (str or Path object).

    Returns:
        A list of dicts, one per page that contains text:
        [{"page_number": int, "text": str}, ...]
        Pages are 1-indexed to match the visible page numbers in a document.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF.
        PDFExtractionError: If PyMuPDF cannot open or read the file.
        EmptyDocumentError: If no text could be extracted from any page.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path.name}")

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        raise PDFExtractionError(f"Could not open PDF: {pdf_path.name}") from exc

    pages = []

    with document:
        for page_index in range(len(document)):
            page = document[page_index]
            text = page.get_text()

            # Skip pages that are blank or contain only whitespace
            if text.strip():
                pages.append({
                    "page_number": page_index + 1,  # Convert to 1-indexed
                    "text": text,
                })

    if not pages:
        raise EmptyDocumentError(
            f"No extractable text found in '{pdf_path.name}'. "
            "The PDF may contain only images or be scanned without OCR."
        )

    return pages
