import chromadb
from sentence_transformers import SentenceTransformer
from ollama import chat


CHROMA_DIR = "chroma_db"
MODEL_NAME = "qwen2.5:3b"


# -----------------------------
# Load embedding model
# -----------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------
# Connect to ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_collection(
    name="government_policies"
)


# -----------------------------
# Retrieve documents
# -----------------------------

def retrieve(query, top_k=10):

    query_embedding = embedding_model.encode(query)

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    retrieved_chunks = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        retrieved_chunks.append({
            "text": document,
            "source": metadata["source"],
            "page": metadata["page"]
        })

    return retrieved_chunks


# -----------------------------
# Build context
# -----------------------------

def build_context(chunks):

    context = ""

    for i, chunk in enumerate(chunks):

        context += f"""
SOURCE {i + 1}
Document: {chunk['source']}
Page: {chunk['page']}

{chunk['text']}

-----------------------------
"""

    return context


# -----------------------------
# Generate answer
# -----------------------------

def generate_answer(query, context, strengthened=None):
    """
    Generate a grounded answer using only retrieved evidence.

    The LLM must:
    - Answer every part of the user's question.
    - Use only information present in the context.
    - Preserve terminology from the source document.
    - Never invent or expand terms that are not defined.
    - Clearly distinguish eligibility requirements from other details.
    """

    # --------------------------------------------------------
    # Use strengthened prompt if provided
    # --------------------------------------------------------

    additional_instruction = ""

    if strengthened:
        additional_instruction = f"""
Additional instructions from the self-healing system:

{strengthened}
"""

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a precise government policy document assistant.

Your task is to answer the user's question using ONLY the
retrieved document context provided below.

IMPORTANT RULES:

1. GROUNDED ANSWERS
   - Every factual statement must be supported by the context.
   - Do not use outside knowledge.
   - Do not guess or infer missing information.

2. ANSWER THE COMPLETE QUESTION
   - Identify every part of the user's question.
   - Answer all parts that are supported by the context.
   - Do not answer only one part when the context contains
     information for multiple parts.

3. PRESERVE SOURCE TERMINOLOGY
   - Use terminology exactly as it appears in the document.
   - Do NOT invent expansions of abbreviations.
   - For example, if the document says "EWS", write "EWS".
   - Do not expand EWS, LIG, MIG, BLC, AHP, ARH, ISS, etc.
     unless the expansion is explicitly present in the context.

4. DO NOT INVENT FACTS
   - Never add information from your general knowledge.
   - Never create eligibility conditions that are not present
     in the retrieved context.

5. DISTINGUISH DIFFERENT CRITERIA
   When relevant, clearly separate:
   - Eligibility requirements
   - Income categories
   - Restrictions/disqualifications
   - Other conditions

6. HANDLE MISSING INFORMATION
   - If one part of the question is not supported by the
     context, answer the supported parts.
   - Then clearly state which part is not available in the
     retrieved context.
   - Do NOT say that the entire answer is unavailable if
     some parts can be answered.

7. NUMERICAL ACCURACY
   - Preserve numbers, currency values, percentages and
     ranges exactly as stated in the source.
   - Do not change ₹3 lakh, ₹6 lakh or ₹9 lakh into different
     values.

8. SOURCE REFERENCES
   - Mention the relevant source and page number when available.
   - Use the source/page information provided in the context.

9. CLARITY
   - Give a direct, concise answer.
   - Use bullet points or a small table when it improves clarity.

USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

{additional_instruction}

Now provide the final answer.
"""

    # --------------------------------------------------------
    # LLM CALL
    # --------------------------------------------------------

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# -----------------------------
# Main
# -----------------------------

def main():

    query = input(
        "\nEnter your question: "
    )

    print("\nRetrieving documents...")

    chunks = retrieve(
        query,
        top_k=10
    )

    context = build_context(chunks)

    print("\nGenerating answer...")

    answer = generate_answer(
        query,
        context
    )

    print("\n")
    print("=" * 70)
    print("BASELINE RAG ANSWER")
    print("=" * 70)

    print(answer)

    print("\n")
    print("=" * 70)
    print("SOURCES USED")
    print("=" * 70)

    for chunk in chunks:

        print(
            f"- {chunk['source']} "
            f"(Page {chunk['page']})"
        )


if __name__ == "__main__":
    main()