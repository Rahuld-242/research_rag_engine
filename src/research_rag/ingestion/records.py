from pathlib import Path

from research_rag.ingestion.ids import get_document_id, make_document_hash
from research_rag.ingestion.metadata import extract_topic_metadata
from research_rag.ingestion.discovery import find_pdf_files
from research_rag.ingestion.extraction import extract_pdf_text
from research_rag.ingestion.chunking import chunk_documents
from research_rag.ingestion.ids import get_chunk_id

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.base import BaseChunk
from docling.document_converter import DocumentConverter

def build_document_record(pdf_path: Path, root_dir: Path) -> dict:
    pdf_path = Path(pdf_path).resolve()
    root_dir = Path(root_dir).resolve()
    
    document_id = get_document_id(pdf_path, root_dir)
    document_hash = make_document_hash(document_id, length=16)
    
    file_name = pdf_path.name
    file_path = str(pdf_path)
    
    topic_metadata = extract_topic_metadata(document_id)
    
    return {
        "document_id" : document_id,
        "document_hash" : document_hash,
        "file_name" : file_name,
        "file_path" : file_path,
        **topic_metadata
    }
    
    
def build_document_records(root_dir: Path) -> list[dict]:
    root_dir = Path(root_dir).resolve()
    pdf_files = find_pdf_files(root_dir)
    
    document_records=[build_document_record(pdf_file, root_dir) for pdf_file in pdf_files]
    
    return document_records
    
def build_extracted_document_record(pdf_path: Path, root_dir: Path) -> dict:
    pdf_path = Path(pdf_path).resolve()
    root_dir = Path(root_dir).resolve()
    
    record = build_document_record(pdf_path, root_dir)
    
    text = extract_pdf_text(pdf_path)
    
    record["text"] = text
    
    return record
    
def build_extracted_document_records(root_dir: Path) -> list[dict]:
    root_dir = Path(root_dir).resolve()
    pdf_files = find_pdf_files(root_dir)
    
    extracted_document_records = [build_extracted_document_record(pdf_file, root_dir) for pdf_file in pdf_files]
    
    return extracted_document_records
    
    
def build_chunk_record(pdf_path: Path, root_dir: Path, chunk: BaseChunk, chunk_index: int, chunker: HybridChunker) -> dict:
    
    pdf_path = Path(pdf_path).resolve()
    root_dir = Path(root_dir).resolve()
    
    chunker_text = chunker.contextualize(chunk)
    meta_rec = build_document_record(pdf_path, root_dir)
    
    meta_rec["chunk_index"] = chunk_index
    meta_rec["chunk_id"] = get_chunk_id(meta_rec["document_id"], chunk_index)
    meta_rec["text"] = chunker_text
            
    return meta_rec

def build_chunk_records(pdf_files: list[Path], root_dir:Path, converter: DocumentConverter, chunker: HybridChunker) -> list[dict]:
    root_dir = Path(root_dir).resolve()
    
    chunked_docs = chunk_documents(pdf_files, converter, chunker)
    
    chunk_meta_recs = []
    
    for pdf_path, chunks in chunked_docs.items():
        for chunk_index, chunk in enumerate(chunks):
            chunk_meta_rec = build_chunk_record(pdf_path, root_dir, chunk, chunk_index, chunker)
            chunk_meta_recs.append(chunk_meta_rec)
    
    return chunk_meta_recs

def get_processed_document_ids(persisted_chunk_records: list[dict])->set[str]:
    processed_documents=set()
    
    for persisted_chunk_record in persisted_chunk_records:
        processed_documents.add(persisted_chunk_record["document_id"])
        
    return processed_documents
        