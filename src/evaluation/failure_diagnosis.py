import json

from ollama import chat


MODEL_NAME = "qwen2.5:3b"


def diagnose_failure(
    question,
    answer,
    context,
    evaluation
):

    prompt = f"""
You are a failure diagnosis component for a
self-healing Retrieval-Augmented Generation system.

Your job is to determine WHY the generated answer failed
and select the most appropriate repair strategy.

QUESTION:
{question}

GENERATED ANSWER:
{answer}

RETRIEVED CONTEXT:
{context}

EVALUATION:
{json.dumps(evaluation, indent=2)}


Choose exactly ONE primary failure type:

1. RETRIEVAL_FAILURE
The required information exists in the knowledge
base, but the retrieved chunks failed to provide
the relevant information.

2. INSUFFICIENT_EVIDENCE
The available context does not contain enough
information to answer the question.

3. GENERATION_FAILURE
The retrieved context contains sufficient information,
but the LLM generated unsupported or incorrect claims.

4. CONFLICTING_EVIDENCE
The retrieved context contains contradictory information.

5. QUERY_ANSWER_MISMATCH
The generated answer or retrieved information is about
a different subject, entity, object, or concept than
the user's question.

Examples:

Question:
"What subsidy is available for agricultural land?"

Retrieved information:
"Housing subsidy for purchasing houses."

This is QUERY_ANSWER_MISMATCH because agricultural land
and housing are different subjects.

Question:
"What are the EWS income limits?"

Answer:
"The ISS loan amount is ₹25 lakh."

This is QUERY_ANSWER_MISMATCH because the answer discusses
loan amount instead of income limits.

This is extremely important.

Examples:

Question:
"What subsidy is available for agricultural land?"

Answer:
"Redeemable Housing Vouchers are available for purchasing
houses."

This is a QUERY_ANSWER_MISMATCH because agricultural land
and housing are different subjects.

Another example:

Question:
"What is the MIG income limit?"

Answer:
"The EWS income limit is ₹3 lakh."

This is also a QUERY_ANSWER_MISMATCH.

Use this failure type when the evaluator reports:

"query_answer_consistency": false


IMPORTANT PRIORITY RULES:

If query_answer_consistency is false,
prefer QUERY_ANSWER_MISMATCH.

If the answer is about the correct subject but contains
unsupported facts, use GENERATION_FAILURE.

If the required information is missing from retrieval,
use INSUFFICIENT_EVIDENCE.

If retrieval contains irrelevant information,
use RETRIEVAL_FAILURE.


Return ONLY valid JSON.

Required format:

{{
    "failure_type": "",
    "confidence": 0.0,
    "reasoning": "",
    "repair_action": ""
}}

For repair_action, choose ONE:

- increase_top_k
- use_query_expansion
- retrieve_again
- strengthen_grounding_prompt
- retrieve_more_authoritative_evidence
- ask_user_for_clarification
"""

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

        result = response["message"]["content"]

        # ----------------------------------------------------
        # Remove markdown code fences if Qwen adds them
        # ----------------------------------------------------

        result = result.strip()

        if result.startswith("```json"):
            result = result[7:]

        elif result.startswith("```"):
            result = result[3:]

        if result.endswith("```"):
            result = result[:-3]

        result = result.strip()

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        return json.loads(result)

    except json.JSONDecodeError:

        return {
            "failure_type": "UNKNOWN",
            "confidence": 0.0,
            "reasoning": "Could not parse diagnosis.",
            "repair_action": "retrieve_again"
        }

    except Exception as e:

        return {
            "failure_type": "UNKNOWN",
            "confidence": 0.0,
            "reasoning": f"Diagnosis failed: {e}",
            "repair_action": "retrieve_again"
        }