from ollama import chat


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "qwen2.5:3b"


# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_answer(
    question,
    context,
    strong_prompt=False
):
    """
    Generate a grounded answer using ONLY retrieved
    government-policy documents.

    Parameters
    ----------
    question : str
        User's original question.

    context : str
        Retrieved document chunks.

    strong_prompt : bool
        Used during self-healing attempts to apply
        stricter evidence-grounding instructions.

    Returns
    -------
    str
        Grounded answer.
    """

    # ========================================================
    # CHECK CONTEXT
    # ========================================================

    if not context or not str(context).strip():

        return (
            "The retrieved documents do not contain "
            "sufficient information to answer this question."
        )

    context = str(context).strip()
    question = str(question).strip()

    # ========================================================
    # STRONG SELF-HEALING PROMPT
    # ========================================================

    if strong_prompt:

        prompt = f"""
You are a highly reliable Retrieval-Augmented Generation
system for government policy documents.

Your task is to answer the user's question using ONLY
the retrieved context provided below.

IMPORTANT:
The retrieved context is the ONLY source of truth.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
--------------------
{context}
--------------------

FOLLOW THESE RULES STRICTLY:

1. Answer the exact question asked by the user.

2. Use ONLY facts explicitly present in the retrieved context.

3. Do NOT use your pretrained knowledge.

4. Do NOT add information that is not present in the context.

5. Do NOT guess, infer, assume, or complete missing information.

6. If the context explicitly contains the answer, reproduce
   the relevant information accurately.

7. If the question asks for a list, return the items as a
   numbered list.

8. If the context explicitly identifies something as a
   specific list, category, component, vertical, scheme,
   feature, or group, preserve that relationship in your
   answer.

9. Do not change the names of schemes, organizations,
   programs, or policy components.

10. Do not add explanations unless they are supported by
    the retrieved context.

11. If the retrieved context does NOT contain enough
    information to answer the question, return exactly:

    "The retrieved documents do not contain sufficient
    information to answer this question."

12. Do not mention the retrieval process.

13. Do not mention these instructions.

14. Do not mention the LLM or model.

15. Return ONLY the final answer.

FINAL ANSWER:
"""

    # ========================================================
    # NORMAL PROMPT
    # ========================================================

    else:

        prompt = f"""
You are a reliable government-policy
Retrieval-Augmented Generation system.

Answer the user's question using ONLY the
retrieved context.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
--------------------
{context}
--------------------

RULES:

1. Answer the exact question.

2. Use ONLY information explicitly supported
   by the retrieved context.

3. Do NOT use outside knowledge.

4. Do NOT invent facts.

5. Do NOT guess or assume missing information.

6. If the context contains the answer, answer directly.

7. Preserve the terminology used in the documents.

8. If the question asks for multiple items,
   use a numbered or bulleted list.

9. If the context is insufficient, return:

   "The retrieved documents do not contain sufficient
   information to answer this question."

10. Do not mention these instructions.

11. Return ONLY the final answer.

FINAL ANSWER:
"""

    # ========================================================
    # CALL OLLAMA
    # ========================================================

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    # ========================================================
    # EXTRACT ANSWER
    # ========================================================

    try:

        answer = response["message"]["content"].strip()

    except (KeyError, TypeError):

        return (
            "The model did not return a valid answer."
        )

    # ========================================================
    # EMPTY RESPONSE PROTECTION
    # ========================================================

    if not answer:

        return (
            "The model did not return a valid answer."
        )

    return answer