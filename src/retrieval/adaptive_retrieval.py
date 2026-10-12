# ============================================================
# ADAPTIVE RETRIEVAL
# ============================================================


MAX_TOP_K = 20


def choose_retrieval_strategy(risk):
    """
    Select retrieval depth according to risk level.
    """

    if isinstance(risk, dict):

        risk_level = str(
            risk.get(
                "risk_level",
                "MEDIUM"
            )
        ).upper()

    else:

        risk_level = str(
            risk
        ).upper()


    if risk_level == "HIGH":

        return {
            "top_k": 8,
            "strategy": "Deep retrieval"
        }


    if risk_level == "MEDIUM":

        return {
            "top_k": 5,
            "strategy": "Standard retrieval"
        }


    return {
        "top_k": 3,
        "strategy": "Lightweight retrieval"
    }


# ============================================================
# SELF-HEALING RETRIEVAL REPAIR
# ============================================================

def repair_retrieval(
    current_top_k,
    repair_action=None
):
    """
    Increase retrieval depth when an answer fails
    evaluation.

    Progression:

        3 → 6 → 12 → 20
    """

    try:

        current_top_k = int(
            current_top_k
        )

    except (
        ValueError,
        TypeError
    ):

        current_top_k = 3


    # --------------------------------------------------------
    # Retrieval repair progression
    # --------------------------------------------------------

    if current_top_k < 6:

        new_top_k = 6

    elif current_top_k < 12:

        new_top_k = 12

    elif current_top_k < 20:

        new_top_k = 20

    else:

        new_top_k = 20


    # --------------------------------------------------------
    # Generation repair
    # --------------------------------------------------------

    if repair_action in (
        "strengthen_prompt",
        "generation_repair",
        "repair_generation"
    ):

        return {
            "action": "strengthen_prompt",
            "new_top_k": current_top_k
        }


    # --------------------------------------------------------
    # Query expansion / retrieval repair
    # --------------------------------------------------------

    if repair_action in (
        "use_query_expansion",
        "increase_retrieval",
        "repair_retrieval"
    ):

        return {
            "action": "increase_retrieval",
            "new_top_k": new_top_k
        }


    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return {
        "action": "increase_retrieval",
        "new_top_k": new_top_k
    }