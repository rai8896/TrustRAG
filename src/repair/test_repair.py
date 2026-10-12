from src.repair.repair import choose_repair


def main():

    test_cases = [
        "RETRIEVAL_FAILURE",
        "INSUFFICIENT_EVIDENCE",
        "GENERATION_FAILURE",
        "CONFLICTING_EVIDENCE",
        "UNKNOWN"
    ]

    for failure_type in test_cases:

        result = choose_repair(
            failure_type,
            5
        )

        print("=" * 60)
        print("Failure Type :", failure_type)
        print("Repair       :", result)


if __name__ == "__main__":
    main()