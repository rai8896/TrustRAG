import re


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "this",
    "that",
    "these",
    "those",
    "there",
    "their",
    "about",
    "which",
    "where",
    "when",
    "what",
    "with",
    "from",
    "into",
    "under",
    "over",
    "than",
    "then",
    "they",
    "them",
    "have",
    "has",
    "does",
    "doesnt",
    "not",
    "only",
    "also",
    "such",
    "provided",
    "available",
    "information",
    "mentioned",
    "context",
    "documents"
}


# ============================================================
# EXTRACT IMPORTANT WORDS
# ============================================================

def extract_keywords(text):
    """
    Extract meaningful words from text.
    """

    words = re.findall(
        r"\b[a-zA-Z]{4,}\b",
        text.lower()
    )

    keywords = []

    for word in words:

        if word in STOP_WORDS:
            continue

        if word not in keywords:
            keywords.append(word)

    return keywords


# ============================================================
# SENTENCE SPLITTER
# ============================================================

def split_sentences(text):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# CHECK EVIDENCE
# ============================================================

def check_evidence(
    answer,
    context
):
    """
    Estimate how strongly the generated answer
    is supported by the retrieved context.

    This is a lightweight lexical checker.

    It is NOT a replacement for semantic evaluation.
    """

    if not answer or not context:

        return {
            "support_score": 0.0,
            "supported_claims": [],
            "unsupported_claims": [
                "Missing answer or context."
            ],
            "evidence_sufficient": False
        }

    context_lower = context.lower()

    sentences = split_sentences(
        answer
    )

    supported = []
    unsupported = []

    for sentence in sentences:

        keywords = extract_keywords(
            sentence
        )

        # ----------------------------------------------------
        # No meaningful keywords
        # ----------------------------------------------------

        if not keywords:

            continue

        # ----------------------------------------------------
        # Count keyword matches
        # ----------------------------------------------------

        matches = 0

        for word in keywords:

            if word in context_lower:

                matches += 1

        support_ratio = (
            matches / len(keywords)
        )

        # ----------------------------------------------------
        # Strong lexical support
        # ----------------------------------------------------

        if support_ratio >= 0.60:

            supported.append(
                sentence
            )

        else:

            unsupported.append(
                sentence
            )

    total = (
        len(supported)
        + len(unsupported)
    )

    if total == 0:

        score = 0.0

    else:

        score = (
            len(supported)
            / total
        )

    return {
        "support_score":
            round(score, 2),

        "supported_claims":
            supported,

        "unsupported_claims":
            unsupported,

        "evidence_sufficient":
            score >= 0.60
    }