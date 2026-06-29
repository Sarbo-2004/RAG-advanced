import os
import re
import fitz
from langchain_core.documents import Document
from src.config import PDF_DIR


def extract_year_from_filename(filename: str) -> str:
    match = re.search(r"(20\d{2})", filename)
    return match.group(1) if match else "unknown"


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_single_pdf(pdf_path: str):
    documents = []
    filename = os.path.basename(pdf_path)
    year = extract_year_from_filename(filename)

    pdf = fitz.open(pdf_path)

    for page_index, page in enumerate(pdf):
        text = page.get_text("text")
        text = clean_text(text)

        if not text:
            continue

        metadata = {
            "source": filename,
            "year": year,
            "page": page_index + 1,
            "content_type": "text"
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata
            )
        )

    pdf.close()
    return documents


def load_all_pdfs(pdf_dir: str = PDF_DIR):
    all_documents = []

    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = [
        file for file in os.listdir(pdf_dir)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        documents = load_single_pdf(pdf_path)
        all_documents.extend(documents)

    return all_documents