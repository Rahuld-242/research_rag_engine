import base64
import hashlib
import uuid
from pathlib import Path

def get_document_id(pdf_path: Path, root_dir: Path) -> str:
    # Resolve both the pdf_path and root_dir
    pdf_path = Path(pdf_path).resolve()
    root_dir = Path(root_dir).resolve()
    
    # Convert the pdf path to relative path with respect to root_dir
    relative_path = pdf_path.relative_to(root_dir)
    
    # Use POSIX format to convert '\' to '/'
    return relative_path.as_posix()

def make_document_hash(document_id: str, length: int=16) -> str:
    # Hash the document id
    raw_bytes = hashlib.sha256(document_id.encode('utf-8')).digest()
    
    # Convert the raw_bytes to reasonable Base32 string
    encoded=base64.b32encode(raw_bytes).decode("ascii")
    
    # Base32 can include "=" padding, so remove it  
    encoded.rstrip("=")
    
    return encoded[:length]
    
def get_chunk_id(document_id: str, chunk_index: int) -> str:
    chunk_key = f"{document_id}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_key))


    
    
    