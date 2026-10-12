import re

from ollama import chat


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "qwen2.5:3b"


# ============================================================
# EXTRACT QUERIES
# ============================================================

def extract_queries(text):
    """
    Extract individual search queries from Qwen's response.

    Handles responses such as:

    1. PMAY-U 2.0 eligibility criteria
    2. PMAY-U 2.0 eligible beneficiaries
    3) PMAY-U 2.0 eligibility requirements
    - PMAY-U 2.0 beneficiary conditions
    """

    lines = text.splitlines()

    queries = []

    for line in lines:

        line = line.strip()

        # ----------------------------------------------------
        # Remove numbering / bullets
        # ----------------------------------------------------

        line = re.sub(
            r"^\s*(?:\d+[\.\)]|[-*])\s*",
            "",
            line
        )

        line = line.strip()

        # ----------------------------------------------------
        # Ignore empty lines
        # ----------------------------------------------------

        if not line:
            continue

        # ----------------------------------------------------
        # Ignore extremely short responses
        # ----------------------------------------------------

        if len(line) < 5:
            continue

        # ----------------------------------------------------
        # Ignore obvious explanations
        # ----------------------------------------------------

        lower_line = line.lower()

        ignored_prefixes = (
            "here are",
            "queries:",
            "alternative queries:",
            "search queries:",
            "the queries are",
        )

        if lower_line.startswith(
            ignored_prefixes
        ):
            continue

        # ----------------------------------------------------
        # Add query
        # ----------------------------------------------------

        queries.append(line)

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    unique_queries = []

    seen = set()

    for query in queries:

        normalized = query.lower().strip()

        if normalized not in seen:

            seen.add(normalized)

            unique_queries.append(query)

    return unique_queries


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(question):
    """
    Generate multiple search queries from the original
    user question.

    The original question is always retained.

    Maximum number of queries = 5.
    """

    prompt = f"""
You are a query expansion component for a
government-policy Retrieval-Augmented Generation system.

The user asked:

{question}

Your task is to generate 4 alternative search queries
that would help retrieve the exact information needed
to answer the user's question from government policy
documents.

RULES:

1. Keep every query focused on the user's question.

2. Use important policy terms from the question.

3. Do NOT answer the question.

4. Do NOT add facts that are not present in the question.

5. Do NOT introduce unrelated topics.

6. Each query should be short and suitable for semantic
   search.

7. The queries should be meaningfully different from
   each other.

8. Return ONLY the queries.

9. Return exactly one query per line.

10. Do not use numbering.

11. Do not use bullet points.

EXAMPLE:

User question:
What are the eligibility criteria for PMAY-U 2.0?

Output:

PMAY-U 2.0 eligibility criteria
PMAY-U 2.0 eligible beneficiaries
PMAY-U 2.0 eligibility requirements
PMAY-U 2.0 beneficiary conditions
"""

    # ========================================================
    # CALL QWEN
    # ========================================================

    try:

        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        result = response[
            "message"
        ][
            "content"
        ]

    except Exception as e:

        print(
            "\nQuery expansion failed:"
        )

        print(e)

        # ----------------------------------------------------
        # Safe fallback:
        # Always use the original question.
        # ----------------------------------------------------

        return [
            question
        ]


    # ========================================================
    # EXTRACT GENERATED QUERIES
    # ========================================================

    queries = extract_queries(
        result
    )


    # ========================================================
    # ALWAYS RETAIN ORIGINAL QUESTION
    # ========================================================

    final_queries = [
        question
    ]


    # ========================================================
    # ADD EXPANDED QUERIES
    # ========================================================

    for query in queries:

        # Avoid adding the exact original question again

        if query.lower().strip() == question.lower().strip():
            continue

        final_queries.append(
            query
        )


    # ========================================================
    # MAXIMUM 5 QUERIES
    # ========================================================

    return final_queries[:5]


# ============================================================
# TEST
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("QUERY EXPANSION TEST")
    print("=" * 70)

    question = (
        "What are the four verticals of PMAY-U 2.0?"
    )

    print(
        f"\nOriginal question:\n{question}"
    )

    print(
        "\nGenerating expanded queries..."
    )

    queries = expand_query(
        question
    )

    print("\n")
    print("=" * 70)
    print("EXPANDED QUERIES")
    print("=" * 70)

    for index, query in enumerate(
        queries,
        start=1
    ):

        print(
            f"{index}. {query}"
        )

    print("\n")
    print(
        f"Total queries: {len(queries)}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()