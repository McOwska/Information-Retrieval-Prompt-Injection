from multihop_agent.agent import format_documents, generate_final_answer


def run_single_shot(
    question: str,
    retriever,
    top_k: int = 5,
) -> dict:
    """
    System B1: single-shot RAG baseline.

    Retrieves top_k documents in one pass and immediately generates the final
    answer. Returns a dict with the same shape as run_multihop_agent so the
    evaluation loop can treat all systems uniformly.

    Args:
        question: input question
        retriever: object with a retrieve(query, top_k) method
        top_k: number of documents to retrieve

    Returns:
        dict with keys: question, answer, hops, all_retrieved_docs
    """
    retrieved_docs = retriever.retrieve(question, top_k=top_k)

    context = format_documents(retrieved_docs)
    answer = generate_final_answer(question=question, context=context)

    return {
        "question": question,
        "answer": answer,
        "hops": [
            {
                "hop": 1,
                "query": question,
                "retrieved_docs": retrieved_docs,
                "flagged_docs": [],
                "new_docs_added": retrieved_docs,
            }
        ],
        "all_retrieved_docs": retrieved_docs,
    }
