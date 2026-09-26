import os
os.environ["DOCLING_INFERENCE_COMPILE_TORCH_MODELS"] = "false"

from time import perf_counter

from pathlib import Path

from research_rag.ingestion.discovery import find_pdf_files

from docling.datamodel.base_models import InputFormat 
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DoclingDocument

pipeline_options = PdfPipelineOptions(
    do_formula_enrichment = True,
    do_ocr = False
)

converter = DocumentConverter(
    format_options = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_options = pipeline_options
        )
    }
)

def parse_pdf(pdf_path: Path, converter: DocumentConverter) -> DoclingDocument:
    result = converter.convert(pdf_path)
    return result.document

def parse_pdfs(pdf_files: list[Path], converter: DocumentConverter) -> dict[Path, DoclingDocument]:
    parsed_pdfs = {}
    for pdf_file in pdf_files:
        start = perf_counter()
        print (f"Parsing: {pdf_file}")
        parsed_pdfs[pdf_file]=parse_pdf(pdf_file, converter)
        print (f"Completed in {perf_counter() - start:.1f}s")
    return parsed_pdfs



