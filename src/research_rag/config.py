from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT=Path(__file__).resolve().parents[2] #Moves up to the parent folder

class Settings(BaseSettings):
    #This will read RAG_SOURCE_DIR from .env and convert it into a Path object
    rag_source_dir: Path
    
    model_config=SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore"    
    )
    
    @property
    def sample_papers_dir(self)->Path:
        #This path if fixed 
        return PROJECT_ROOT / "raw_data" / "sample_papers"
    
#Creating one reusable settings object for the whole project
settings=Settings()