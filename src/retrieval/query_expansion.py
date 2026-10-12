import json
import re

from ollama import chat


MODEL_NAME = "qwen2.5:3b"


def extract_json(text):

    text = text.strip()

    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    )

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        text = text[
            start:end + 1
        ]

    return text.strip()


def expand_query(query):

    prompt = f"""
You are a query expansion system for a
government policy RAG system.

Generate alternative search queries that
can help retrieve relevant government policy
documents.

Do NOT answer the question.

USER QUESTION:

{query}

Generate exactly 4 alternative queries.

Return ONLY valid JSON:

{{
    "queries": [
        "alternative query 1",
        "alternative query 2",
        "alternative query 3",
        "alternative query 4"
    ]
}}
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response[
        "message"
    ][
        "content"
    ]

    try:

        cleaned = extract_json(
            content
        )

        data = json.loads(
            cleaned
        )

        queries = data.get(
            "queries",
            []
        )

        if not isinstance(
            queries,
            list
        ):

            queries = []

        # Keep only strings

        queries = [
            q.strip()
            for q in queries
            if isinstance(q, str)
            and q.strip()
        ]

        # Original question FIRST

        final_queries = [
            query
        ]

        for q in queries:

            if q.lower() != query.lower():

                final_queries.append(q)

        return final_queries[:5]

    except Exception:

        # Safe fallback

        return [
            query
        ]