from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


PDF_PATH = "data/pdfs/2023_pdf.pdf"

DOCLING_MODELS_PATH = r"D:\OneDrive - Coforge Limited\Desktop\RAG-advanced\docling_models"


def main():
    print("Testing Docling with local models and OCR disabled...")

    pipeline_options = PdfPipelineOptions(
        artifacts_path=DOCLING_MODELS_PATH,
        do_ocr=False,
        do_table_structure=False
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    result = converter.convert(PDF_PATH)

    markdown_text = result.document.export_to_markdown()

    print("Docling conversion successful.")
    print("=" * 100)
    print(markdown_text[:3000])


if __name__ == "__main__":
    main()