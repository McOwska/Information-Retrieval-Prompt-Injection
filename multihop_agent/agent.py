from llm import call_llm


def format_documents(docs: list[dict], max_chars_per_doc: int = 1200) -> str:
    """
    Converts retrieved documents into a readable context for the LLM.
    """
    formatted_docs = []

    for i, doc in enumerate(docs, start=1):
        title = doc.get("title", "Untitled")
        text = doc.get("text", "")

        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "..."

        formatted_docs.append(
            f"[{i}] Title: {title}\n{text}"
        )

    return "\n\n".join(formatted_docs)


def generate_followup_query(
    question: str,
    context_so_far: str,
    hop: int,
) -> str:
    short_context = context_so_far[:1500]

    messages = [
        {
            "role": "system",
            "content": (
                "Return only a short keyword search query. "
                "Do not explain. Do not write a full sentence."
            ),
        },
        {
            "role": "user",
            "content": f"""
Question:
{question}

Context:
{short_context}

Next search query:
""".strip(),
        },
    ]

    query = call_llm(
        messages=messages,
        temperature=0.0,
        max_tokens=128,
    )

    query = clean_query(query)

    if not query:
        return question

    return query


def clean_query(query: str) -> str:
    """
    Removes unnecessary formatting from LLM-generated queries.
    """
    query = query.strip()

    if query.startswith('"') and query.endswith('"'):
        query = query[1:-1]

    if query.startswith("'") and query.endswith("'"):
        query = query[1:-1]

    return query.strip()


def generate_final_answer(question: str, context: str) -> str:
    """
    Uses the LLM to answer the question based on all retrieved documents.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are a question answering system. "
                "Answer the question using only the provided context. "
                "If the answer cannot be found in the context, say \"I don't know\". "
                "Answer the question in the simplest and shortest way, idealy just one-word/phrase/piece of information."
            ),
        },
        {
            "role": "user",
            "content": f"""
Question:
{question}

Context:
{context}

Answer:
""".strip(),
        },
    ]

    return call_llm(
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )


def add_new_documents(
    all_docs: list[dict],
    retrieved_docs: list[dict],
    seen_doc_ids: set[str],
) -> list[dict]:
    """
    Adds only documents that have not been added before.
    Returns the newly added documents.
    """
    new_docs = []

    for doc in retrieved_docs:
        doc_id = doc.get("doc_id") or doc.get("title")

        if doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            all_docs.append(doc)
            new_docs.append(doc)

    return new_docs


def apply_classifier_guard(
    docs: list[dict],
    classifier,
) -> tuple[list[dict], list[dict]]:
    """
    Splits retrieved docs into clean and flagged lists using the classifier.

    The classifier must be callable as classifier(text: str) -> bool,
    returning True when the document is considered adversarial.

    Args:
        docs: list of retrieved document dicts
        classifier: callable that takes document text and returns True if adversarial

    Returns:
        (clean_docs, flagged_docs)
    """
    clean_docs = []
    flagged_docs = []

    for doc in docs:
        text = doc.get("text", "")
        if classifier(text):
            flagged_docs.append(doc)
        else:
            clean_docs.append(doc)

    return clean_docs, flagged_docs


def run_multihop_agent(
    question: str,
    retriever,
    max_hops: int = 3,
    top_k: int = 3,
    classifier=None,
    poisoned_hops: list[int] = None,
    poisoned_retriever=None,
) -> dict:
    """
    Runs a simple multi-hop retrieval agent.

    Args:
        question: input question
        retriever: object with retrieve(query, top_k) method
        max_hops: number of retrieval steps
        top_k: number of documents retrieved at each step
        classifier: optional callable(text: str) -> bool. When provided, each
                    retrieved chunk is screened before entering context (S1 system).
                    When None, the agent runs without a guard (B2 system).

    Returns:
        dict with final answer and logs from each hop
    """

    hops = []
    all_retrieved_docs = []
    seen_doc_ids = set()

    current_query = question

    for hop in range(1, max_hops + 1):
        
        if poisoned_hops and poisoned_retriever and hop in poisoned_hops:
            retrieved_docs = poisoned_retriever.retrieve(current_query, top_k=top_k)

        retrieved_docs = retriever.retrieve(current_query, top_k=top_k)

        flagged_docs: list[dict] = []
        if classifier is not None:
            retrieved_docs, flagged_docs = apply_classifier_guard(retrieved_docs, classifier)

        new_docs = add_new_documents(
            all_docs=all_retrieved_docs,
            retrieved_docs=retrieved_docs,
            seen_doc_ids=seen_doc_ids,
        )

        context_so_far = format_documents(all_retrieved_docs)

        hops.append({
            "hop": hop,
            "query": current_query,
            "retrieved_docs": retrieved_docs,
            "flagged_docs": flagged_docs,
            "new_docs_added": new_docs,
        })

        if hop < max_hops:
            current_query = generate_followup_query(
                question=question,
                context_so_far=context_so_far,
                hop=hop + 1,
            )

    final_context = format_documents(all_retrieved_docs)
    final_answer = generate_final_answer(
        question=question,
        context=final_context,
    )

    return {
        "question": question,
        "answer": final_answer,
        "hops": hops,
        "all_retrieved_docs": all_retrieved_docs,
    }