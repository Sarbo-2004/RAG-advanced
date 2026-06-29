"""Citation formatting for retrieved Docling chunks.

Kept dependency-free (operates on LangChain ``Document`` objects only) so it is
trivially unit-testable without loading models or the vector store.
"""

from typing import Iterable, List

from langchain_core.documents import Document


# Section headings that carry no useful locating information for a citation.
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
    """True when a section heading is too generic to add to a citation."""
    if not section:
        return True
    return clean_section_for_citation(section).lower() in _GENERIC_SECTIONS


def format_citations(docs: Iterable[Document]) -> List[str]:
    """Build a de-duplicated, human-readable citation list from documents.

    Each citation is ``"<source>, Page <page>"`` optionally suffixed with a
    meaningful section heading. Order of first appearance is preserved.
    """
    citation_map = {}

    for doc in docs:
        metadata = doc.metadata
        source = metadata.get("source", "Unknown source")
        page = metadata.get("page", "Unknown page")
        section = metadata.get("section", "")
        content_type = metadata.get("content_type", "")

        cleaned_section = clean_section_for_citation(section)
        key = f"{source}_page_{page}_{cleaned_section}_{content_type}"

        if cleaned_section and not is_generic_section(section):
            citation = f"{source}, Page {page}, Section: {cleaned_section}"
        else:
            citation = f"{source}, Page {page}"

        citation_map.setdefault(key, citation)

    return list(citation_map.values())
