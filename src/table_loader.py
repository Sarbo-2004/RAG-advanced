import os
import re
import pdfplumber
from langchain_core.documents import Document
from src.config import PDF_DIR


def extract_year_from_filename(filename: str) -> str:
    match = re.search(r"(20\d{2})", filename)
    return match.group(1) if match else "unknown"


def clean_cell(cell):
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip()


def clean_row(row):
    return [clean_cell(cell) for cell in row]


def extract_table_title_from_page(page_text: str):
    """
    Extract actual table title from page text.
    Only accepts lines that start with TABLE <number>.
    """

    if not page_text:
        return None

    lines = page_text.split("\n")

    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()

        if re.match(r"^TABLE\s+\d+", line, re.IGNORECASE):
            return line

    return None


def is_probable_real_table(table):
    """
    Basic structural filter to remove empty/single-column/very small false tables.
    """

    if not table:
        return False

    cleaned_rows = [clean_row(row) for row in table]
    cleaned_rows = [row for row in cleaned_rows if any(row)]

    if len(cleaned_rows) < 2:
        return False

    max_cols = max(len(row) for row in cleaned_rows)

    if max_cols < 2:
        return False

    non_empty_cells = 0
    total_cells = 0

    for row in cleaned_rows:
        for cell in row:
            total_cells += 1
            if cell:
                non_empty_cells += 1

    if total_cells == 0:
        return False

    fill_ratio = non_empty_cells / total_cells

    if fill_ratio < 0.25:
        return False

    return True


def table_to_text(table, source, page_number, table_title):
    cleaned_rows = []

    for row in table:
        cleaned_row = clean_row(row)
        if any(cleaned_row):
            cleaned_rows.append(cleaned_row)

    if not cleaned_rows:
        return ""

    header = cleaned_rows[0]
    data_rows = cleaned_rows[1:]

    text_parts = [
        f"Table Title: {table_title}",
        f"Source: {source}",
        f"Page: {page_number}",
        "Content Type: Table",
        "",
        "The table content is represented below in a readable row-wise format."
    ]

    for row_index, row in enumerate(data_rows, start=1):
        row_items = []

        for col_index, value in enumerate(row):
            if not value:
                continue

            column_name = (
                header[col_index]
                if col_index < len(header) and header[col_index]
                else f"Column {col_index + 1}"
            )

            row_items.append(f"{column_name}: {value}")

        if row_items:
            text_parts.append(f"Row {row_index}: " + " | ".join(row_items))

    return "\n".join(text_parts)


def load_tables_from_single_pdf(pdf_path: str):
    documents = []

    filename = os.path.basename(pdf_path)
    year = extract_year_from_filename(filename)

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1

            try:
                page_text = page.extract_text() or ""
                tables = page.extract_tables()
            except Exception:
                tables = []

            if not tables:
                continue

            table_title = extract_table_title_from_page(page_text)

            # Important:
            # If no real TABLE title is found on the page, skip all extracted pseudo-tables.
            if not table_title:
                continue

            for table_index, table in enumerate(tables, start=1):
                if not is_probable_real_table(table):
                    continue

                table_text = table_to_text(
                    table=table,
                    source=filename,
                    page_number=page_number,
                    table_title=table_title
                )

                if not table_text.strip():
                    continue

                metadata = {
                    "source": filename,
                    "year": year,
                    "page": page_number,
                    "content_type": "table",
                    "table_title": table_title,
                    "table_id": f"{filename}_page_{page_number}_table_{table_index}"
                }

                documents.append(
                    Document(
                        page_content=table_text,
                        metadata=metadata
                    )
                )

    return documents


def load_all_tables(pdf_dir: str = PDF_DIR):
    all_table_documents = []

    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = [
        file for file in os.listdir(pdf_dir)
        if file.lower().endswith(".pdf")
    ]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        table_docs = load_tables_from_single_pdf(pdf_path)
        all_table_documents.extend(table_docs)

    return all_table_documents