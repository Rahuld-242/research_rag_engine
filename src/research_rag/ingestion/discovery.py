from pathlib import Path
from research_rag.ingestion.ids import get_document_id

def find_pdf_files(root_dir: Path) -> list[Path]:
    
    #Convert string to Path Object
    root_dir= Path(root_dir)
    
    #Check if directory exists
    if not root_dir.exists():
        raise FileNotFoundError(f"The root directory does not exist: {root_dir}")
    
    #Check if root path is a directory
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_dir}")
    
    pdf_files=[]
    
    #Recursively appending files in folders and subfolders
    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower()==".pdf":
            pdf_files.append(path)
            
    return sorted(pdf_files)

def find_unprocessed_pdf_files(root_dir: Path, processed_document_ids: set[str]) -> list[Path]:

    root_dir = Path(root_dir).resolve()
    
    pdf_files = find_pdf_files(root_dir)
    
    unprocessed_pdf_files = []
    
    for pdf_file in pdf_files:
        document_id = get_document_id(pdf_file, root_dir)
        if document_id not in processed_document_ids:
            unprocessed_pdf_files.append(pdf_file)
            
    return unprocessed_pdf_files
        
        
        
    
    
    
        
    
    