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


def is_markdown_table_block(block: str) -> bool:
    """
    Detects whether a block looks like a Markdown table.
    """
    lines = block.strip().splitlines()

    if len(lines) < 2:
        return False

    pipe_lines = [line for line in lines if "|" in line]

    return len(pipe_lines) >= 2


def is_table_of_contents_like(text: str) -> bool:
    """
    Detects content/contents pages that list section titles and page numbers.
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
        "page"
    ]

    keyword_count = sum(1 for keyword in toc_keywords if keyword in lower_text)

    figure_count = len(re.findall(r"\bfigure\s+\d+", lower_text))
    table_count = len(re.findall(r"\btable\s+\d+", lower_text))

    # If a chunk has many figure/table references and section names, it is likely TOC.
    if keyword_count >= 4 and (figure_count >= 3 or table_count >= 2):
        return True

    # Markdown TOC tables often have many pipes and page-number references.
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
        "food and agriculture organization of the united nations"
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
    Splits Markdown text into sections based on headings like:
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
                        "content": "\n".join(current_content).strip()
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
                "content": "\n".join(current_content).strip()
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
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return splitter.split_text(text)


def split_docling_documents(docling_documents):
    """
    Converts Docling page-level Markdown documents into RAG-friendly chunks.

    Strategy:
    - Remove image placeholders
    - Skip cover/citation/contents chunks
    - Split Markdown by headings
    - Keep Markdown tables as complete chunks where useful
    - Split only large text sections
    - Preserve metadata
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

            if is_markdown_table_block(content):
                metadata["content_type"] = "docling_table_markdown"

                final_chunks.append(
                    Document(
                        page_content=content,
                        metadata=metadata
                    )
                )

                chunk_counter += 1

            else:
                small_chunks = split_large_text(content)

                for small_chunk in small_chunks:
                    small_chunk = remove_markdown_image_tags(small_chunk)

                    if is_low_value_section(small_chunk, heading):
                        continue

                    chunk_metadata = metadata.copy()
                    chunk_metadata["content_type"] = "docling_text"
                    chunk_metadata["chunk_id"] = chunk_counter

                    final_chunks.append(
                        Document(
                            page_content=small_chunk,
                            metadata=chunk_metadata
                        )
                    )

                    chunk_counter += 1

    return final_chunks