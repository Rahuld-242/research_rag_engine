from research_rag.ingestion.ids import make_document_hash
from research_rag.ingestion.ids import get_document_id

def test_make_document_hash_is_deterministic():
    document_id="05_RAG_and_Retrieval/RAG Survey.pdf"
    
    hash1=make_document_hash(document_id)
    hash2=make_document_hash(document_id)
    
    assert hash1==hash2
    
def test_make_document_hash_respects_length():
    document_id="05_RAG_and_Retrieval/RAG Survey.pdf"
    
    document_hash=make_document_hash(document_id, length=16)
    
    assert(len(document_hash))==16
    
def test_make_document_hash_has_no_padding():
    document_id="05_RAG_and_Retrieval/RAG Survey.pdf"
    
    document_hash=make_document_hash(document_id, length=16)
    
    assert "=" not in document_hash
    
def test_get_document_id_return_posix_relative_path(tmp_path):
    root_dir=tmp_path/"AIML Research Papers"
    topic_dir=root_dir/"05_RAG_and_Retrieval"
    topic_dir.mkdir(parents=True)
    
    pdf_path=topic_dir/"RAG Survey.pdf"
    pdf_path.write_text("dummy pdf content")
    
    document_id=get_document_id(pdf_path, root_dir)
    
    assert document_id=="05_RAG_and_Retrieval/RAG Survey.pdf"