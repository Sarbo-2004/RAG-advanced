"""Unit tests for citation formatting (no models / API required)."""

from langchain_core.documents import Document

from src.citations import (
    clean_section_for_citation,
    format_citations,
    is_generic_section,
)


def _doc(**metadata) -> Document:
    return Document(page_content="x", metadata=metadata)


def test_clean_section_strips_markdown_and_whitespace():
    assert clean_section_for_citation("  ## True Cost Accounting  ") == "True Cost Accounting"
    assert clean_section_for_citation("") == ""


def test_generic_sections_are_detected():
    assert is_generic_section("Untitled Section") is True
    assert is_generic_section("## the state of food and agriculture 2023 in brief") is True
    assert is_generic_section("True Cost Accounting") is False
    assert is_generic_section("") is True


def test_citation_includes_meaningful_section():
    docs = [_doc(source="2023_pdf.pdf", page=22, section="## True Cost Accounting")]
    assert format_citations(docs) == ["2023_pdf.pdf, Page 22, Section: True Cost Accounting"]


def test_citation_omits_generic_section():
    docs = [_doc(source="2021_pdf.pdf", page=7, section="Untitled Section")]
    assert format_citations(docs) == ["2021_pdf.pdf, Page 7"]


def test_citations_are_deduplicated_and_ordered():
    docs = [
        _doc(source="2023_pdf.pdf", page=4, section="Core Messages"),
        _doc(source="2023_pdf.pdf", page=4, section="Core Messages"),
        _doc(source="2025_pdf.pdf", page=1, section="Untitled Section"),
    ]
    assert format_citations(docs) == [
        "2023_pdf.pdf, Page 4, Section: Core Messages",
        "2025_pdf.pdf, Page 1",
    ]
