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


def format_citations(docs: Iterable[Document], question: str = "") -> List[str]:
    """Build a clean, de-duplicated citation list from documents."""

    docs = list(docs)

    if not docs:
        return []

    question_lower = question.lower()

    asks_for_visual = any(
        term in question_lower
        for term in ["figure", "chart", "map", "diagram", "visual"]
    )

    filtered_docs = []

    for doc in docs:
        section = clean_section_for_citation(
            doc.metadata.get("section", "")
        ).lower()

        content_type = doc.metadata.get("content_type", "").lower()

        is_figure = (
            "figure" in section
            or "figure" in content_type
            or "chart" in section
            or "map" in section
        )

        if is_figure and not asks_for_visual:
            continue

        filtered_docs.append(doc)

    if not filtered_docs:
        filtered_docs = docs

    citation_map = {}

    for doc in filtered_docs:
        metadata = doc.metadata

        source = metadata.get("source", "Unknown source")
        page = metadata.get("page", "Unknown page")
        section = metadata.get("")
        table_title = metadata.get("table_title", "")
        content_type = metadata.get("content_type", "")

        cleaned_section = clean_section_for_citation(section)
        cleaned_table_title = clean_section_for_citation(table_title)

        key = f"{source}_page_{page}_{cleaned_table_title or cleaned_section}_{content_type}"

        if cleaned_table_title:
            citation = f"{source}, Page {page}, {cleaned_table_title}"
        elif cleaned_section and not is_generic_section(section):
            citation = f"{source}, Page {page}, Section: {cleaned_section}"
        else:
            citation = f"{source}, Page {page}"

        citation_map.setdefault(key, citation)

    return list(citation_map.values())