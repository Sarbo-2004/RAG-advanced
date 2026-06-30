import os
import re

import fitz
from langchain_core.documents import Document

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.config import DOCLING_MODELS_PATH, PDF_DIR


def extract_year_from_filename(filename: str) -> str:
    match = re.search(r"(20\d{2})", filename)
    return match.group(1) if match else "unknown"


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_pdf_page_count(pdf_path: str) -> int:
    with fitz.open(pdf_path) as pdf:
        return pdf.page_count


def create_docling_converter(do_table_structure: bool = True):
    pipeline_options = PdfPipelineOptions(
        artifacts_path=DOCLING_MODELS_PATH,
        do_ocr=False,
        do_table_structure=do_table_structure,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    return converter


def extract_page_with_pymupdf(pdf_path: str, page_number: int) -> str:
    with fitz.open(pdf_path) as pdf:
        page = pdf[page_number - 1]
        text = page.get_text("text")

    return clean_text(text)


def convert_single_page_with_docling(
    pdf_path: str,
    page_number: int,
    do_table_structure: bool = True,
) -> str:
    converter = create_docling_converter(
        do_table_structure=do_table_structure
    )

    result = converter.convert(
        pdf_path,
        page_range=(page_number, page_number),
    )

    markdown_text = result.document.export_to_markdown()

    return markdown_text.strip()


def load_single_pdf_pagewise_docling(pdf_path: str):
    documents = []

    filename = os.path.basename(pdf_path)
    year = extract_year_from_filename(filename)
    total_pages = get_pdf_page_count(pdf_path)

    print(f"\nProcessing PDF with page-wise Docling: {filename}")
    print(f"Total pages: {total_pages}")

    for page_number in range(1, total_pages + 1):
        print(f"Processing page {page_number}/{total_pages}...")

        page_content = ""
        parser_used = ""
        table_structure_used = False
        extraction_status = "success"

        try:
            page_content = convert_single_page_with_docling(
                pdf_path=pdf_path,
                page_number=page_number,
                do_table_structure=True,
            )

            parser_used = "docling"
            table_structure_used = True

        except Exception as table_error:
            print(f"Docling table mode failed on page {page_number}: {table_error}")
            print("Trying Docling without table structure...")

            try:
                page_content = convert_single_page_with_docling(
                    pdf_path=pdf_path,
                    page_number=page_number,
                    do_table_structure=False,
                )

                parser_used = "docling"
                table_structure_used = False
                extraction_status = "docling_without_table_structure"

            except Exception as docling_error:
                print(f"Docling text mode failed on page {page_number}: {docling_error}")
                print("Falling back to PyMuPDF...")

                try:
                    page_content = extract_page_with_pymupdf(
                        pdf_path=pdf_path,
                        page_number=page_number,
                    )

                    parser_used = "pymupdf_fallback"
                    table_structure_used = False
                    extraction_status = "pymupdf_fallback"

                except Exception as fallback_error:
                    print(f"PyMuPDF fallback failed on page {page_number}: {fallback_error}")
                    continue

        if not page_content.strip():
            continue

        metadata = {
            "source": filename,
            "year": year,
            "page": page_number,
            "content_type": "docling_page",
            "parser": parser_used,
            "table_structure": table_structure_used,
            "chunk_source": "docling_pagewise",
            "extraction_status": extraction_status,
        }

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata,
            )
        )

    return documents


def load_all_pdfs_pagewise_docling(pdf_dir: str = PDF_DIR):
    all_documents = []

    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = sorted(
        file for file in os.listdir(pdf_dir)
        if file.lower().endswith(".pdf")
    )

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)

        docs = load_single_pdf_pagewise_docling(pdf_path)
        all_documents.extend(docs)

    return all_documents