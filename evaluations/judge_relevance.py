from typing import Any
from evaluations.schemas import RelevanceJudgement

from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate

def judge_relevance(
    query: str,
    chunk: str,
    llm: Runnable[Any, RelevanceJudgement]
) -> RelevanceJudgement:
    system_message= """
        You are a retrieval relevance evaluator.
        Evaluate how much useful evidence the retrieved chunk contributes toward answering the specific query. Consider factual evidence, explanations, comparisons, and supporting context. Do not assign relevance based solely on topical similarity.
        The grading is described as:
        0: No useful evidence for answering the query
        1: Partial or supporting evidence
        2: Direct or essential answer bearing evidence
        """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "Query: {query}\n\nChunk: {chunk}")
        ]
    )
    
    chain = prompt|llm
    
    result = chain.invoke(
        {
            "query": query,
            "chunk": chunk
        }
    )
    
    return result

    