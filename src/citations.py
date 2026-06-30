"""Citation formatting for retrieved Docling chunks.

This module works only on LangChain Document objects.
It does not load models, vector stores, or external dependencies.
"""

from typing import Iterable, List

from langchain_core.documents import Document


# Section headings that carry no useful locating information.
_GENERIC_SECTIONS = {
    "untitled section",
    "the state of food and agriculture 2021 in brief",
    "the state of food and agriculture 2022 in brief",
    "the state of food and agriculture 2023 in brief",
    "the state of food and agriculture 2024 in brief",
    "the state of food and agriculture 2025 in brief",
    "required citation:",
    "food and agriculture the state of",
}


def clean_section_for_citation(section: str) -> str:
    """Strip Markdown heading markers and surrounding whitespace."""

    if not section:
        return ""

    return section.strip().lstrip("#").strip()


def is_generic_section(section: str) -> bool:
    """Return True if a section heading is too generic for citation."""

    if not section:
        return True

    cleaned = clean_section_for_citation(section).lower()

    return cleaned in _GENERIC_SECTIONS


def is_table_document(doc: Document) -> bool:
    """Return True if a document looks like a table chunk."""

    metadata = doc.metadata

    content_type = metadata.get("content_type", "").lower()
    section = metadata.get("section", "").lower()
    table_title = metadata.get("table_title", "").lower()

    if "table" in content_type:
        return True

    if "table" in section:
        return True

    if "table" in table_title:
        return True

    return False


def build_single_citation(doc: Document) -> str:
    """Build one formatted citation string."""

    metadata = doc.metadata

    source = metadata.get("source", "Unknown source")
    page = metadata.get("page", "Unknown page")
    section = metadata.get("section", "")
    table_title = metadata.get("table_title", "")

    cleaned_section = clean_section_for_citation(section)
    cleaned_table_title = clean_section_for_citation(table_title)

    # Prefer table title if available.
    if cleaned_table_title:
        return f"{source}, Page {page}, {cleaned_table_title}"

    # Then prefer meaningful section.
    if cleaned_section and not is_generic_section(section):
        return f"{source}, Page {page}, Section: {cleaned_section}"

    # Fallback to page-only citation.
    return f"{source}, Page {page}"


def format_citations(docs: Iterable[Document]) -> List:
    """Build a clean, de-duplicated citation list.

    Logic:
    - If table chunks are present, prefer table citations.
    - De-duplicate citations.
    - Preserve first-seen order.
    """

    docs = list(docs)

    if not docs:
        return []

    table_docs = [doc for doc in docs if is_table_document(doc)]

    # If table chunks are retrieved, citation should focus on table chunks.
    # This prevents extra figure/text pages from polluting exact table-value answers.
    citation_source_docs = table_docs if table_docs else docs

    citation_map = {}

    for doc in citation_source_docs:
        metadata = doc.metadata

        source = metadata.get("source", "Unknown source")
        page = metadata.get("page", "Unknown page")
        section = clean_section_for_citation(metadata.get("section", ""))
        table_title = clean_section_for_citation(metadata.get("table_title", ""))
        content_type = metadata.get("content_type", "")

        # If table is present on same page, page-level duplicates are avoided.
        key = f"{source}_page_{page}_{table_title or section}_{content_type}"

        citation = build_single_citation(doc)

        if citation not in citation_map.values():
            citation_map[key] = citation

    return list(citation_map.values())