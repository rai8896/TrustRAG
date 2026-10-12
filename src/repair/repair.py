def choose_repair(failure_type, current_top_k):
    """
    Select a repair strategy based on the diagnosed failure.
    """

    if failure_type == "RETRIEVAL_FAILURE":

        new_top_k = min(current_top_k * 2, 20)

        return {
            "action": "increase_top_k",
            "new_top_k": new_top_k
        }

    elif failure_type == "INSUFFICIENT_EVIDENCE":

        new_top_k = min(current_top_k * 2, 20)

        return {
            "action": "retrieve_again",
            "new_top_k": new_top_k
        }

    elif failure_type == "QUERY_ANSWER_MISMATCH":

        # The retrieved/generated answer does not match
        # the actual subject of the question.
        #
        # Therefore we need better retrieval rather
        # than simply strengthening the generation prompt.

        new_top_k = min(current_top_k * 2, 20)

        return {
            "action": "use_query_expansion",
            "new_top_k": new_top_k
        }

    elif failure_type == "GENERATION_FAILURE":

        return {
            "action": "strengthen_prompt",
            "new_top_k": current_top_k
        }

    elif failure_type == "CONFLICTING_EVIDENCE":

        new_top_k = min(current_top_k * 2, 20)

        return {
            "action": "retrieve_more_evidence",
            "new_top_k": new_top_k
        }

    else:

        new_top_k = min(current_top_k * 2, 20)

        return {
            "action": "retrieve_again",
            "new_top_k": new_top_k
        }