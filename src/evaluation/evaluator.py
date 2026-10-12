import json
import re

from ollama import chat


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "qwen2.5:3b"


# ============================================================
# EXTRACT JSON
# ============================================================

def extract_json(text):
    """
    Extract a JSON object from the LLM response.

    Handles:
    - normal JSON
    - ```json ... ```
    - ``` ... ```
    """

    text = text.strip()

    # Remove markdown code fences
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

    # Locate JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        text = text[
            start:end + 1
        ]

    return text.strip()


# ============================================================
# NORMALIZE BOOLEAN
# ============================================================

def normalize_boolean(value):
    """
    Convert different LLM representations into boolean.
    """

    if isinstance(value, bool):
        return value

    if isinstance(value, str):

        value = value.strip().lower()

        if value in {
            "true",
            "yes",
            "supported",
            "pass"
        }:
            return True

        if value in {
            "false",
            "no",
            "unsupported",
            "fail"
        }:
            return False

    return False


# ============================================================
# NORMALIZE SCORE
# ============================================================

def normalize_score(value):
    """
    Safely convert a score into the range 0-1.
    """

    try:

        score = float(value)

    except (
        ValueError,
        TypeError
    ):

        score = 0.0

    return max(
        0.0,
        min(
            1.0,
            score
        )
    )


# ============================================================
# EVALUATE ANSWER
# ============================================================

def evaluate_answer(
    question,
    answer,
    context
):
    """
    Evaluate a generated RAG answer.

    The evaluator checks:

    1. Relevance
    2. Faithfulness
    3. Evidence sufficiency
    4. Query-answer consistency
    5. Unsupported claims
    6. Answer type

    IMPORTANT:

    The evaluator must NOT mark an answer as unsupported
    merely because the retrieved context uses slightly
    different wording.

    For example:

    Question:
        What are the four verticals of PMAY-U 2.0?

    Context:
        PMAY-U 2.0 will be implemented through four verticals:
        BLC, AHP, ARH and ISS.

    Answer:
        The four verticals are BLC, AHP, ARH and ISS.

    This should be considered SUPPORTED.
    """

    prompt = f"""
You are a strict evaluation component of a
self-healing Retrieval-Augmented Generation (RAG) system.

Your task is to evaluate whether the generated answer
is supported by the retrieved context.

DO NOT answer the user's question.

============================================================
USER QUESTION
============================================================

{question}


============================================================
GENERATED ANSWER
============================================================

{answer}


============================================================
RETRIEVED CONTEXT
============================================================

{context}


============================================================
EVALUATION CRITERIA
============================================================

1. RELEVANCE
------------

Determine whether the answer addresses the user's question.

Score:

1.0 = directly answers the question
0.5 = partially addresses the question
0.0 = does not address the question


2. FAITHFULNESS
---------------

Determine whether the claims in the answer are supported
by the retrieved context.

Score:

1.0 = claims are supported by the context
0.5 = partially supported
0.0 = unsupported or contradicted


3. EVIDENCE SUFFICIENCY
-----------------------

Determine whether the retrieved context contains enough
information to answer the user's question.

IMPORTANT:

Do NOT require the context to use exactly the same wording
as the answer.

Semantic equivalence is acceptable.

For example:

CONTEXT:
"PMAY-U 2.0 will be implemented through the following
four verticals: BLC, AHP, ARH and ISS."

ANSWER:
"The four verticals of PMAY-U 2.0 are BLC, AHP, ARH and ISS."

This IS sufficient evidence.

The wording is different, but the meaning is the same.

Therefore:

evidence_sufficiency = true


4. QUERY-ANSWER CONSISTENCY
---------------------------

Determine whether the answer actually answers
the specific question asked.

For example:

QUESTION:
"What are the four verticals of PMAY-U 2.0?"

ANSWER:
"PMAY-U provides housing subsidies."

This is NOT consistent because the answer does not
provide the four verticals.

However:

QUESTION:
"What are the four verticals of PMAY-U 2.0?"

ANSWER:
"BLC, AHP, ARH and ISS."

This IS consistent.


5. UNSUPPORTED CLAIMS
---------------------

Identify claims in the answer that cannot be supported
by the retrieved context.

IMPORTANT:

Do NOT classify a claim as unsupported merely because:

- the wording differs
- the answer summarizes the context
- the answer converts a sentence into a list
- the answer uses abbreviations
- the answer rearranges information already present
- the answer directly answers the question using information
  that is clearly present in the context

Semantic equivalence is allowed.


6. ANSWER TYPE
--------------

Choose exactly ONE:

SUPPORTED
----------
The answer directly addresses the question and is supported
by the retrieved context.

PARTIALLY_SUPPORTED
-------------------
The answer is partially supported, but some required
information is missing.

UNSUPPORTED
-----------
The answer contains claims that are not supported by
the retrieved context.

SAFE_ABSTENTION
---------------
The answer explicitly says that the available evidence
is insufficient, and the context genuinely does not
contain enough information to answer the question.

IMPORTANT:

Do NOT classify an answer as SAFE_ABSTENTION if the context
actually contains the requested information.

Do NOT classify a correct answer as UNSUPPORTED merely
because the context uses different wording.


============================================================
SPECIAL RULE FOR LIST QUESTIONS
============================================================

For questions asking for:

- lists
- categories
- verticals
- components
- types
- stages
- steps
- requirements
- objectives

compare the semantic content of the list.

Example:

QUESTION:
"What are the four verticals of PMAY-U 2.0?"

CONTEXT:
"PMAY-U 2.0 will be implemented through:
Beneficiary Led Construction (BLC),
Affordable Housing in Partnership (AHP),
Affordable Rental Housing (ARH), and
Interest Subsidy Scheme (ISS)."

ANSWER:
"The four verticals are:
1. BLC
2. AHP
3. ARH
4. ISS"

RESULT:

relevance = 1.0
faithfulness = 1.0
evidence_sufficiency = true
query_answer_consistency = true
answer_type = "SUPPORTED"
overall = "GOOD"


============================================================
IMPORTANT EVALUATION RULE
============================================================

Evaluate the MEANING of the context, not exact wording.

The answer may:

- summarize
- paraphrase
- restructure
- enumerate
- abbreviate
- directly extract

information from the context.

These are acceptable as long as the information
is supported by the context.


============================================================
OVERALL RESULT
============================================================

Return:

"GOOD"

ONLY when:

- relevance >= 0.5
- faithfulness >= 0.5
- evidence_sufficiency is true
- query_answer_consistency is true
- answer_type is SUPPORTED

Otherwise return:

"BAD"


============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
    "relevance": 1.0,
    "faithfulness": 1.0,
    "evidence_sufficiency": true,
    "query_answer_consistency": true,
    "unsupported_claims": [],
    "answer_type": "SUPPORTED",
    "reasoning": "",
    "overall": "GOOD"
}}
"""


    # ========================================================
    # CALL QWEN
    # ========================================================

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


    # ========================================================
    # PARSE RESPONSE
    # ========================================================

    try:

        cleaned_result = extract_json(
            result
        )

        data = json.loads(
            cleaned_result
        )


        # ----------------------------------------------------
        # NORMALIZE VALUES
        # ----------------------------------------------------

        relevance = normalize_score(
            data.get(
                "relevance",
                0.0
            )
        )

        faithfulness = normalize_score(
            data.get(
                "faithfulness",
                0.0
            )
        )

        evidence_sufficiency = normalize_boolean(
            data.get(
                "evidence_sufficiency",
                False
            )
        )

        query_answer_consistency = normalize_boolean(
            data.get(
                "query_answer_consistency",
                False
            )
        )


        # ----------------------------------------------------
        # UNSUPPORTED CLAIMS
        # ----------------------------------------------------

        unsupported_claims = data.get(
            "unsupported_claims",
            []
        )

        if not isinstance(
            unsupported_claims,
            list
        ):

            unsupported_claims = [
                str(
                    unsupported_claims
                )
            ]


        # ----------------------------------------------------
        # ANSWER TYPE
        # ----------------------------------------------------

        answer_type = data.get(
            "answer_type",
            "UNSUPPORTED"
        )

        if not isinstance(
            answer_type,
            str
        ):

            answer_type = "UNSUPPORTED"

        answer_type = answer_type.upper().strip()


        valid_answer_types = {
            "SUPPORTED",
            "PARTIALLY_SUPPORTED",
            "UNSUPPORTED",
            "SAFE_ABSTENTION"
        }

        if answer_type not in valid_answer_types:

            answer_type = "UNSUPPORTED"


        # ----------------------------------------------------
        # REASONING
        # ----------------------------------------------------

        reasoning = data.get(
            "reasoning",
            ""
        )

        if not isinstance(
            reasoning,
            str
        ):

            reasoning = str(
                reasoning
            )


        # ====================================================
        # DETERMINE OVERALL RESULT
        # ====================================================

        overall = "BAD"


        if (
            relevance >= 0.5
            and faithfulness >= 0.5
            and evidence_sufficiency
            and query_answer_consistency
            and answer_type == "SUPPORTED"
        ):

            overall = "GOOD"


        # ----------------------------------------------------
        # Return normalized evaluation
        # ----------------------------------------------------

        return {

            "relevance": relevance,

            "faithfulness": faithfulness,

            "evidence_sufficiency":
                evidence_sufficiency,

            "query_answer_consistency":
                query_answer_consistency,

            "unsupported_claims":
                unsupported_claims,

            "answer_type":
                answer_type,

            "reasoning":
                reasoning,

            "overall":
                overall
        }


    # ========================================================
    # HANDLE INVALID MODEL OUTPUT
    # ========================================================

    except Exception as e:

        print(
            "\nEvaluator returned invalid output:"
        )

        print(result)

        print(
            "\nEvaluation parsing error:"
        )

        print(e)


        # ----------------------------------------------------
        # Safe fallback
        # ----------------------------------------------------

        return {

            "relevance": 0.0,

            "faithfulness": 0.0,

            "evidence_sufficiency": False,

            "query_answer_consistency": False,

            "unsupported_claims": [
                "Evaluator returned invalid output."
            ],

            "answer_type":
                "UNSUPPORTED",

            "reasoning":
                "The evaluator failed to produce valid JSON.",

            "overall":
                "BAD"
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("EVALUATOR TEST")
    print("=" * 70)


    question = (
        "What are the four verticals "
        "of PMAY-U 2.0?"
    )


    context = """
    PMAY-U 2.0 will be implemented through
    the following four verticals:

    Beneficiary Led Construction (BLC)

    Affordable Housing in Partnership (AHP)

    Affordable Rental Housing (ARH)

    Interest Subsidy Scheme (ISS)
    """


    answer = """
    The four verticals of PMAY-U 2.0 are:

    1. Beneficiary Led Construction (BLC)
    2. Affordable Housing in Partnership (AHP)
    3. Affordable Rental Housing (ARH)
    4. Interest Subsidy Scheme (ISS)
    """


    evaluation = evaluate_answer(
        question,
        answer,
        context
    )


    print("\nEVALUATION:")
    print(
        json.dumps(
            evaluation,
            indent=4
        )
    )