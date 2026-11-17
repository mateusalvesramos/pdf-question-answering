"""
Unit tests for the pdf_extractor module.

Test strategy:
- Use the sample PDF in docs/ for realistic tests (provided by the user).
- Use pytest's tmp_path fixture to create edge-case files dynamically.
- Never rely on network access or external services.
"""

from pathlib import Path

import fitz
import pytest

from src.pdf_extractor import (
    EmptyDocumentError,
    PDFExtractionError,
    extract_pages,
)

SAMPLE_PDF = Path(__file__).parent.parent / "docs" / "sample.pdf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pdf_with_text(path: Path, pages_text: list[str]) -> None:
    """Create a minimal PDF with the given text on each page."""
    document = fitz.open()
    for text in pages_text:
        page = document.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    document.save(path)


def make_blank_pdf(path: Path, num_pages: int = 1) -> None:
    """Create a PDF with blank pages (no extractable text)."""
    document = fitz.open()
    for _ in range(num_pages):
        document.new_page()
    document.save(path)


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------

class TestExtractPagesSuccess:

    def test_returns_list(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        make_pdf_with_text(pdf, ["Hello world"])
        result = extract_pages(pdf)
        assert isinstance(result, list)

    def test_single_page_content(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        make_pdf_with_text(pdf, ["Hello world"])
        result = extract_pages(pdf)
        assert len(result) == 1
        assert "Hello world" in result[0]["text"]

    def test_page_number_is_one_indexed(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        make_pdf_with_text(pdf, ["Page one content"])
        result = extract_pages(pdf)
        assert result[0]["page_number"] == 1

    def test_multiple_pages_count(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        make_pdf_with_text(pdf, ["Page 1", "Page 2", "Page 3"])
        result = extract_pages(pdf)
        assert len(result) == 3

    def test_multiple_pages_sequential_numbers(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        make_pdf_with_text(pdf, ["First", "Second", "Third"])
        result = extract_pages(pdf)
        page_numbers = [p["page_number"] for p in result]
        assert page_numbers == [1, 2, 3]

    def test_dict_has_expected_keys(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        make_pdf_with_text(pdf, ["Some text"])
        result = extract_pages(pdf)
        assert set(result[0].keys()) == {"page_number", "text"}

    def test_accepts_string_path(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        make_pdf_with_text(pdf, ["Text"])
        # Should not raise even when pdf_path is a str, not a Path
        result = extract_pages(str(pdf))
        assert len(result) == 1

    def test_sample_pdf_is_readable(self):
        """Integration-style: validates that the sample PDF in docs/ can be extracted."""
        if not SAMPLE_PDF.exists():
            pytest.skip("Place a PDF file at docs/sample.pdf to run this test.")
        result = extract_pages(SAMPLE_PDF)
        assert len(result) >= 1

    def test_sample_pdf_pages_have_expected_structure(self):
        """Each page from the sample PDF must have 'page_number' and non-empty 'text'."""
        if not SAMPLE_PDF.exists():
            pytest.skip("Place a PDF file at docs/sample.pdf to run this test.")
        result = extract_pages(SAMPLE_PDF)
        for page in result:
            assert "page_number" in page
            assert "text" in page
            assert isinstance(page["page_number"], int)
            assert len(page["text"].strip()) > 0


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestExtractPagesErrors:

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_pages(tmp_path / "nonexistent.pdf")

    def test_raises_value_error_for_non_pdf(self, tmp_path):
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("not a pdf")
        with pytest.raises(ValueError, match="not a PDF"):
            extract_pages(txt_file)

    def test_raises_pdf_extraction_error_for_corrupt_file(self, tmp_path):
        corrupt = tmp_path / "corrupt.pdf"
        corrupt.write_bytes(b"this is not valid pdf content at all")
        with pytest.raises(PDFExtractionError):
            extract_pages(corrupt)

    def test_raises_empty_document_error_for_blank_pdf(self, tmp_path):
        pdf = tmp_path / "blank.pdf"
        make_blank_pdf(pdf, num_pages=2)
        with pytest.raises(EmptyDocumentError):
            extract_pages(pdf)

    def test_blank_pages_are_skipped(self, tmp_path):
        """A PDF with one blank page and one text page should return only the text page."""
        document = fitz.open()

        blank_page = document.new_page()  # noqa: F841 — intentionally left blank

        text_page = document.new_page()
        text_page.insert_text((72, 72), "Real content on page 2", fontsize=12)

        pdf = tmp_path / "mixed.pdf"
        document.save(pdf)

        result = extract_pages(pdf)

        assert len(result) == 1
        assert result[0]["page_number"] == 2
        assert "Real content" in result[0]["text"]
