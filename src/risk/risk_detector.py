import json
import re

from ollama import chat


MODEL_NAME = "qwen2.5:3b"


# ============================================================
# EXTRACT JSON FROM MODEL RESPONSE
# ============================================================

def extract_json(text):
    """
    Extract a JSON object from the model response.

    Handles responses where Qwen puts JSON inside
    markdown code fences.
    """

    text = text.strip()

    # Remove ```json
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove ```
    text = re.sub(
        r"```\s*",
        "",
        text
    )

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        text = text[
            start:end + 1
        ]

    return text.strip()


# ============================================================
# RISK DETECTION
# ============================================================

def detect_risk(question):

    prompt = f"""
You are a risk classification system for a
government policy RAG system.

Your task is to classify the user's question
into LOW, MEDIUM, or HIGH risk.

Do NOT answer the question.

============================================================
USER QUESTION
============================================================

{question}


============================================================
RISK DEFINITIONS
============================================================

LOW RISK
---------

General informational questions.

These questions usually ask for basic facts
about a government policy.

Examples:

- What is PMAY-U 2.0?
- When was PMAY-U 2.0 launched?
- What are the objectives of PMAY-U 2.0?


MEDIUM RISK
-----------

Questions involving specific policy rules,
requirements, eligibility conditions, income
limits, documents, deadlines, or benefits.

Examples:

- What are the eligibility criteria?
- What is the income limit for EWS?
- What documents are required?
- What are the benefits of the scheme?


HIGH RISK
---------

Questions involving personal eligibility,
legal consequences, financial decisions,
or decisions that could significantly affect
the user.

Examples:

- Am I personally eligible for PMAY-U 2.0?
- Can I legally claim this benefit?
- Will I lose my subsidy if I sell the house?
- Can I apply if I already own a house?
- Will I face a penalty if I do this?


============================================================
RISK SCORE
============================================================

Assign a numerical risk_score between 0 and 1.

LOW:
0.00 - 0.33

MEDIUM:
0.34 - 0.66

HIGH:
0.67 - 1.00


IMPORTANT:

The numerical risk_score MUST be consistent
with the risk_level.

For example:

risk_score = 0.20
risk_level = LOW

risk_score = 0.50
risk_level = MEDIUM

risk_score = 0.80
risk_level = HIGH


Do not answer the user's question.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
    "risk_level": "LOW",
    "risk_score": 0.20,
    "reasoning": "The question asks for general information."
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
        # Get risk score
        # ----------------------------------------------------

        try:

            risk_score = float(
                data.get(
                    "risk_score",
                    0.5
                )
            )

        except (
            ValueError,
            TypeError
        ):

            risk_score = 0.5


        # ----------------------------------------------------
        # Keep score within 0-1
        # ----------------------------------------------------

        risk_score = max(
            0.0,
            min(
                1.0,
                risk_score
            )
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Make the numerical score authoritative.
        #
        # This prevents an inconsistent result such as:
        #
        # risk_level = MEDIUM
        # risk_score = 0.67
        #
        # because 0.67 belongs to HIGH.
        # ----------------------------------------------------

        if risk_score <= 0.33:

            risk_level = "LOW"

        elif risk_score <= 0.66:

            risk_level = "MEDIUM"

        else:

            risk_level = "HIGH"


        # ----------------------------------------------------
        # Reasoning
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


        # ----------------------------------------------------
        # Return normalized result
        # ----------------------------------------------------

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "reasoning": reasoning
        }


    # ========================================================
    # HANDLE INVALID MODEL OUTPUT
    # ========================================================

    except Exception as e:

        print(
            "\nRisk detector returned invalid output:"
        )

        print(result)

        print(
            "\nError:"
        )

        print(e)


        # ----------------------------------------------------
        # Safe fallback
        # ----------------------------------------------------

        return {
            "risk_level": "MEDIUM",
            "risk_score": 0.50,
            "reasoning": (
                "Risk classification failed. "
                "The system defaulted to MEDIUM risk."
            )
        }