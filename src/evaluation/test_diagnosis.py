from src.evaluation.failure_diagnosis import diagnose_failure


question = """
What are the eligibility criteria under PMAY-U 2.0?
"""


answer = """
Anyone living in India can receive PMAY-U 2.0 benefits.
Applicants must be Indian citizens for at least 10 years.
Applicants must also own less than two houses.
The government provides ₹10 lakh to every beneficiary.
"""


context = """
EWS households have annual income up to ₹3 lakh.
LIG households have annual income from ₹3 lakh
up to ₹6 lakh.
MIG households have annual income from ₹6 lakh
up to ₹9 lakh.
"""


evaluation = {
    "relevance": 0.0,
    "faithfulness": 0.0,
    "evidence_sufficiency": False,
    "unsupported_claims": [
        "Anyone living in India can receive PMAY-U 2.0 benefits.",
        "Applicants must be Indian citizens for at least 10 years.",
        "Applicants must also own less than two houses."
    ],
    "reasoning": "The context does not provide enough information.",
    "overall": "BAD"
}


result = diagnose_failure(
    question,
    answer,
    context,
    evaluation
)


print("\n" + "=" * 60)
print("FAILURE DIAGNOSIS")
print("=" * 60)

print(result)