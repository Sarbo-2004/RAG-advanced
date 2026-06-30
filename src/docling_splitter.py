import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


MIN_CHUNK_LENGTH = 120


def remove_markdown_image_tags(text: str) -> str:
    """
    Removes Docling image placeholder tags like <!-- image -->.
    """

    text = re.sub(r"<!--\s*image\s*-->", "", text, flags=re.IGNORECASE)
    return text.strip()


def clean_heading_text(heading: str) -> str:
    """
    Removes Markdown heading symbols from a section title.
    """

    if not heading:
        return "Untitled Section"

    return heading.strip().lstrip("#").strip()


def is_table_heading(heading: str) -> bool:
    """
    Detects whether a section heading is a real table heading.
    """

    heading_clean = clean_heading_text(heading).lower()

    return bool(re.search(r"\btable\s+\d+", heading_clean))


def is_figure_heading(heading: str) -> bool:
    """
    Detects figure headings so they are not confused with tables.
    """

    heading_clean = clean_heading_text(heading).lower()

    return bool(re.search(r"\bfigure\s+\d+", heading_clean))


def is_markdown_table_block(block: str) -> bool:
    """
    Detects whether a block looks like a Markdown table.

    We require:
    - at least two pipe lines
    - preferably one separator line containing --- 
    """

    lines = block.strip().splitlines()

    if len(lines) < 2:
        return False

    pipe_lines = [line for line in lines if "|" in line]

    if len(pipe_lines) < 2:
        return False

    has_separator = any(
        set(line.strip()) <= set("|-: ") and "-" in line
        for line in pipe_lines
    )

    return has_separator or len(pipe_lines) >= 3


def is_table_of_contents_like(text: str) -> bool:
    """
    Detects contents pages that list section titles and page numbers.
    These pages are usually not useful for RAG answers.
    """

    lower_text = text.lower()

    toc_keywords = [
        "contents",
        "core messages",
        "foreword",
        "summary",
        "figure",
        "table",
        "page",
    ]

    keyword_count = sum(1 for keyword in toc_keywords if keyword in lower_text)

    figure_count = len(re.findall(r"\bfigure\s+\d+", lower_text))
    table_count = len(re.findall(r"\btable\s+\d+", lower_text))

    if keyword_count >= 4 and (figure_count >= 3 or table_count >= 2):
        return True

    if "|" in text and keyword_count >= 4 and figure_count >= 2:
        return True

    return False


def is_low_value_section(text: str, heading: str) -> bool:
    """
    Skips chunks that are unlikely to help answer user questions.
    """

    cleaned_text = remove_markdown_image_tags(text)
    lower_text = cleaned_text.lower()
    lower_heading = heading.lower()

    if not cleaned_text.strip():
        return True

    if len(cleaned_text.strip()) < MIN_CHUNK_LENGTH:
        return True

    skip_phrases = [
        "required citation",
        "cover photograph",
        "some rights reserved",
        "isbn",
        "sales, rights and licensing",
        "third-party materials",
        "food and agriculture organization of the united nations",
    ]

    if any(phrase in lower_text for phrase in skip_phrases):
        return True

    if "required citation" in lower_heading:
        return True

    if is_table_of_contents_like(cleaned_text):
        return True

    return False


def split_markdown_by_headings(markdown_text: str):
    """
    Splits Markdown text into sections based on headings:
    # Heading
    ## Heading
    ### Heading
    """

    markdown_text = remove_markdown_image_tags(markdown_text)

    lines = markdown_text.splitlines()
    sections = []

    current_heading = "Untitled Section"
    current_content = []

    for line in lines:
        stripped_line = line.strip()

        if re.match(r"^#{1,6}\s+", stripped_line):
            if current_content:
                sections.append(
                    {
                        "heading": current_heading,
                        "content": "\n".join(current_content).strip(),
                    }
                )

            current_heading = stripped_line
            current_content = [stripped_line]

        else:
            current_content.append(line)

    if current_content:
        sections.append(
            {
                "heading": current_heading,
                "content": "\n".join(current_content).strip(),
            }
        )

    return [section for section in sections if section["content"]]


def split_large_text(text: str):
    """
    Splits large non-table text blocks using RecursiveCharacterTextSplitter.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_text(text)


def build_contextual_chunk_text(content: str, metadata: dict) -> str:
    """
    Adds lightweight context prefix before the actual chunk.
    This improves retrieval and reranking quality.
    """

    source = metadata.get("source", "Unknown source")
    year = metadata.get("year", "Unknown year")
    page = metadata.get("page", "Unknown page")
    section = clean_heading_text(metadata.get("section", "Untitled Section"))
    content_type = metadata.get("content_type", "unknown")

    prefix = (
        f"Document: {source}\n"
        f"Year: {year}\n"
        f"Page: {page}\n"
        f"Section: {section}\n"
        f"Content Type: {content_type}\n\n"
    )

    return prefix + content.strip()


def split_docling_documents(docling_documents):
    """
    Converts Docling page-level Markdown documents into RAG-friendly chunks.

    Strategy:
    - Remove image placeholders.
    - Skip cover/citation/contents chunks.
    - Split Markdown by headings.
    - Preserve table sections completely.
    - Preserve Markdown tables completely.
    - Split only large normal text sections.
    - Add rich metadata.
    """

    final_chunks = []
    chunk_counter = 1

    for doc in docling_documents:
        base_metadata = doc.metadata.copy()
        page_content = remove_markdown_image_tags(doc.page_content)

        if not page_content.strip():
            continue

        sections = split_markdown_by_headings(page_content)

        for section in sections:
            heading = section["heading"]
            content = remove_markdown_image_tags(section["content"])

            if is_low_value_section(content, heading):
                continue

            metadata = base_metadata.copy()
            metadata["section"] = heading
            metadata["chunk_source"] = "docling_markdown"
            metadata["chunk_id"] = chunk_counter

            table_heading = is_table_heading(heading)
            markdown_table = is_markdown_table_block(content)
            figure_heading = is_figure_heading(heading)

            # --------------------------------------------------
            # Preserve actual table sections completely
            # --------------------------------------------------
            if table_heading or (markdown_table and not figure_heading):
                metadata["content_type"] = "docling_table_markdown"
                metadata["chunk_id"] = chunk_counter

                chunk_text = build_contextual_chunk_text(
                    content=content,
                    metadata=metadata,
                )

                final_chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=metadata,
                    )
                )

                chunk_counter += 1
                continue

            # --------------------------------------------------
            # Preserve figure sections as figure text
            # --------------------------------------------------
            if figure_heading:
                metadata["content_type"] = "docling_figure_text"
                metadata["chunk_id"] = chunk_counter

                chunk_text = build_contextual_chunk_text(
                    content=content,
                    metadata=metadata,
                )

                final_chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=metadata,
                    )
                )

                chunk_counter += 1
                continue

            # --------------------------------------------------
            # Normal text section splitting
            # --------------------------------------------------
            small_chunks = split_large_text(content)

            for small_chunk in small_chunks:
                small_chunk = remove_markdown_image_tags(small_chunk)

                if is_low_value_section(small_chunk, heading):
                    continue

                chunk_metadata = metadata.copy()
                chunk_metadata["content_type"] = "docling_text"
                chunk_metadata["chunk_id"] = chunk_counter

                chunk_text = build_contextual_chunk_text(
                    content=small_chunk,
                    metadata=chunk_metadata,
                )

                final_chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=chunk_metadata,
                    )
                )

                chunk_counter += 1

    return final_chunks