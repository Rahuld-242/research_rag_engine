
from evaluations.llm import create_judge_llm
from evaluations.judge_relevance import judge_relevance

# Case 10 from our human audit: q5, rank 4
query = (
    "How does Advanced RAG address the retrieval "
    "limitations of Naive RAG?"
)

chunk = """
One of the primary challenges with Naive RAG is its direct
reliance on the user's original query as the basis for retrieval.
Formulating a precise and clear question is difficult, and
imprudent queries result in subpar retrieval effectiveness.

Sometimes, the question itself is complex, and the language
is not well-organized.

Query Expansion: Expanding a single query into multiple queries
enriches the content of the query, providing further context
to address any lack of specific nuances, thereby ensuring
the optimal relevance of the generated answers.

Multi-Query: By employing prompt engineering to expand queries
via LLMs, these queries can then be executed in parallel.
"""

llm = create_judge_llm("gpt-oss:20b")

judgement = judge_relevance(query, chunk, llm)

print("\n=== V2 Relevance Judge ===")
print(f"Query: {query}")
print(f"Grade: {judgement.grade}")
print(f"Reason:\n{judgement.reason}")
