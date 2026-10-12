from src.risk.risk_detector import detect_risk


questions = [
    "What is PMAY-U 2.0?",
    "What are the eligibility criteria under PMAY-U 2.0?",
    "Am I personally eligible for PMAY-U 2.0?",
    "Can I legally claim the PMAY-U 2.0 subsidy if I already own a house?"
]


for question in questions:

    print("\n" + "=" * 70)

    print("QUESTION:")
    print(question)

    result = detect_risk(question)

    print("\nRISK RESULT:")
    print(result)