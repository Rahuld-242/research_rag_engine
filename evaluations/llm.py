from typing import Any

from evaluations.schemas import RelevanceJudgement

from langchain_ollama import ChatOllama
from langchain_core.runnables import Runnable

def create_judge_llm(model_name: str) -> Runnable[Any, RelevanceJudgement]:
    llm = ChatOllama(
        model = model_name,
        temperature = 0
    )
    
    structured_llm = llm.with_structured_output(RelevanceJudgement)
    
    return structured_llm