from typing import Any
from evaluations.schemas import RelevanceJudgement

from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate

def judge_relevance(
    query: str,
    chunk: str,
    llm: Runnable[Any, RelevanceJudgement]
) -> RelevanceJudgement:
    system_message = """
        You are a retrieval relevance evaluator.

        Your task is to evaluate how much useful evidence a retrieved chunk
        provides toward answering the SPECIFIC user query.

        Evaluate the chunk as evidence, not as a complete generated answer.

        RELEVANCE GRADES:

        0 - NO USEFUL EVIDENCE
            The chunk provides no substantive information that helps answer
            the query. Topical similarity, unrelated discussion, and
            bibliographic references alone are insufficient.

        1 - SUPPORTING EVIDENCE
            The chunk provides useful background, definitions, context,
            problem descriptions, or supporting facts relevant to the query,
            but does not directly establish the requested explanation,
            mechanism, comparison, or finding.

        2 - DIRECT ANSWER-BEARING EVIDENCE
            The chunk explicitly provides an important fact, mechanism,
            comparison, or finding that directly answers at least one
            substantive part of the query.

            A grade of 2 does NOT require the chunk to answer the entire
            query or provide a complete explanation.

        EVALUATION RULES:

        1. Identify what the query specifically asks for.
           Distinguish questions asking for problems, mechanisms,
           comparisons, results, or definitions.

        2. Evaluate only the information present in the retrieved chunk.
           Do not fill gaps using external knowledge or assumptions.

        3. Distinguish supporting evidence from direct evidence.
           For example, describing a problem without explaining its
           solution is supporting evidence when the query asks
           how the problem is solved.

        4. Do not confuse evidence relevance with answer completeness.
           A chunk explaining one requested mechanism can receive
           grade 2 even if other mechanisms are missing.

        5. Do not assign grade 2 based on topical similarity or
           technical terminology alone. The chunk must substantively
           address a requested part of the query.

        6. If the chunk contains useful evidence but lacks the
           explanation or connection requested by the query,
           prefer grade 1 over grade 2.

       JUSTIFICATION:

       In the reason field, explain:
        - Query requirement: What information is being requested?
        - Evidence present: What relevant evidence does the chunk contain?
        - Relationship: Why does this evidence justify the assigned grade?
        - Missing evidence: What important information is absent, if any?

       Keep the justification concise and grounded in the chunk.
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

    