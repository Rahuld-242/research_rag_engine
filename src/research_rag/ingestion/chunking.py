from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.base import BaseChunk

from research_rag.ingestion.docling_parser import parse_pdfs

chunker = HybridChunker(
    merge_peers = True
)

def chunk_document(doc: DoclingDocument, chunker: HybridChunker) -> list[BaseChunk]:
    chunks = list(chunker.chunk(dl_doc = doc))
    return chunks

def chunk_documents(pdf_files: list[Path], converter: DocumentConverter, chunker: HybridChunker) -> dict[Path,list[BaseChunk]]:
    parsed_pdfs = parse_pdfs(pdf_files, converter)
    chunked_docs = {pdf_path: chunk_document(parsed_pdf, chunker) for pdf_path, parsed_pdf in parsed_pdfs.items()}
    return chunked_docs


        
        
        
    
