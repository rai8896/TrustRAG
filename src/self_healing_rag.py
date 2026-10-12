"""
SELF-HEALING RAG SYSTEM
=======================

Modes:

1. Terminal mode:
       python -m src.self_healing_rag

2. API / Frontend mode:
       from src.self_healing_rag import run_self_healing_rag

Pipeline:

Question
   ↓
Risk Detection
   ↓
Adaptive Retrieval
   ↓
Query Expansion
   ↓
Vector Retrieval
   ↓
Answer Generation
   ↓
LLM Evaluation
   ↓
Evidence Checking
   ↓
Failure Diagnosis
   ↓
Self-Healing Repair
   ↓
Final Answer
"""

from pathlib import Path
import sys
import traceback


# ============================================================
# PROJECT ROOT
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

# D:\Self-healing rag
PROJECT_ROOT = CURRENT_FILE.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.risk.risk_detector import detect_risk

from src.query_expander import expand_query

from src.retrieval.adaptive_retrieval import (
    choose_retrieval_strategy,
    repair_retrieval
)

from src.retrieval.search import search

from src.evaluation.evaluator import evaluate_answer

from src.evaluation.evidence_checker import check_evidence

from src.evaluation.failure_diagnosis import diagnose_failure


# ============================================================
# GENERATOR IMPORT
# ============================================================

# Your project currently does not have a separate
# src/generator.py visible in the project structure.
#
# baseline_rag.py is therefore used as the fallback location
# for generate_answer.
#
# If your baseline_rag.py exposes generate_answer(), this
# import will work directly.

try:

    from src.retrieval.baseline_rag import generate_answer

except ImportError:

    generate_answer = None


# ============================================================
# CONFIGURATION
# ============================================================

MAX_REPAIR_ATTEMPTS = 3

DEFAULT_TOP_K = 3

MAX_TOP_K = 20


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(
    value,
    default=0.0
):
    """
    Safely convert a value to float.
    """

    try:

        return float(value)

    except (
        ValueError,
        TypeError
    ):

        return default


# ------------------------------------------------------------

def safe_bool(
    value,
    default=False
):
    """
    Safely convert a value to boolean.
    """

    if isinstance(
        value,
        bool
    ):

        return value

    if isinstance(
        value,
        str
    ):

        value = value.lower().strip()

        if value in (
            "true",
            "yes",
            "1"
        ):

            return True

        if value in (
            "false",
            "no",
            "0"
        ):

            return False

    return default


# ------------------------------------------------------------

def make_json_safe(value):
    """
    Convert dictionaries/lists/objects into JSON-safe values.
    """

    if value is None:

        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool
        )
    ):

        return value

    if isinstance(
        value,
        dict
    ):

        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple
        )
    ):

        return [
            make_json_safe(item)
            for item in value
        ]

    try:

        return str(value)

    except Exception:

        return None


# ============================================================
# SEARCH RESULT NORMALIZATION
# ============================================================

def normalize_search_result(
    results
):
    """
    Normalize different possible search() return formats.

    Supported formats:

    FORMAT 1
    --------
    {
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{...}, {...}]],
        "distances": [[0.1, 0.2]]
    }

    FORMAT 2
    --------
    {
        "documents": ["doc1", "doc2"],
        "metadatas": [{...}, {...}],
        "distances": [0.1, 0.2]
    }

    FORMAT 3
    --------
    [
        {
            "text": "...",
            "source": "...",
            "page": 1
        },
        ...
    ]

    FORMAT 4
    --------
    [
        "document 1",
        "document 2"
    ]

    The rest of the pipeline always receives:

    {
        "documents": [...],
        "metadatas": [...],
        "distances": [...]
    }
    """

    # --------------------------------------------------------
    # Empty result
    # --------------------------------------------------------

    if results is None:

        return {
            "documents": [],
            "metadatas": [],
            "distances": []
        }


    # ========================================================
    # CASE 1: DICTIONARY
    # ========================================================

    if isinstance(
        results,
        dict
    ):

        raw_documents = results.get(
            "documents",
            []
        )

        raw_metadatas = results.get(
            "metadatas",
            []
        )

        raw_distances = results.get(
            "distances",
            []
        )


        # ----------------------------------------------------
        # Normalize documents
        # ----------------------------------------------------

        if (
            isinstance(
                raw_documents,
                list
            )
            and len(raw_documents) > 0
            and isinstance(
                raw_documents[0],
                list
            )
        ):

            documents = raw_documents[0]

        else:

            documents = raw_documents


        # ----------------------------------------------------
        # Normalize metadata
        # ----------------------------------------------------

        if (
            isinstance(
                raw_metadatas,
                list
            )
            and len(raw_metadatas) > 0
            and isinstance(
                raw_metadatas[0],
                list
            )
        ):

            metadatas = raw_metadatas[0]

        else:

            metadatas = raw_metadatas


        # ----------------------------------------------------
        # Normalize distances
        # ----------------------------------------------------

        if (
            isinstance(
                raw_distances,
                list
            )
            and len(raw_distances) > 0
            and isinstance(
                raw_distances[0],
                list
            )
        ):

            distances = raw_distances[0]

        else:

            distances = raw_distances


        if not isinstance(
            documents,
            list
        ):

            documents = []


        if not isinstance(
            metadatas,
            list
        ):

            metadatas = []


        if not isinstance(
            distances,
            list
        ):

            distances = []


        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances
        }


    # ========================================================
    # CASE 2: LIST
    # ========================================================

    if isinstance(
        results,
        list
    ):

        documents = []

        metadatas = []

        distances = []


        for item in results:

            # ------------------------------------------------
            # Dictionary document
            # ------------------------------------------------

            if isinstance(
                item,
                dict
            ):

                text = item.get(
                    "text",
                    item.get(
                        "document",
                        item.get(
                            "content",
                            ""
                        )
                    )
                )

                metadata = item.get(
                    "metadata",
                    {}
                )

                # Some search functions put metadata
                # directly inside the item.

                if not metadata:

                    metadata = {
                        "source": item.get(
                            "source",
                            "Unknown"
                        ),
                        "page": item.get(
                            "page",
                            "Unknown"
                        ),
                        "chunk_id": item.get(
                            "chunk_id",
                            "Unknown"
                        )
                    }


                distance = item.get(
                    "distance",
                    None
                )


                if text:

                    documents.append(
                        str(text)
                    )

                    metadatas.append(
                        metadata
                    )

                    distances.append(
                        distance
                    )


            # ------------------------------------------------
            # Plain string document
            # ------------------------------------------------

            elif isinstance(
                item,
                str
            ):

                if item.strip():

                    documents.append(
                        item
                    )

                    metadatas.append(
                        {}
                    )

                    distances.append(
                        None
                    )


        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances
        }


    # ========================================================
    # UNKNOWN FORMAT
    # ========================================================

    return {
        "documents": [],
        "metadatas": [],
        "distances": []
    }


# ============================================================
# EXTRACT RETRIEVED DOCUMENTS
# ============================================================

def extract_documents(
    results
):
    """
    Convert normalized search results into a list of
    structured document objects.
    """

    normalized = normalize_search_result(
        results
    )

    documents = normalized.get(
        "documents",
        []
    )

    metadatas = normalized.get(
        "metadatas",
        []
    )

    distances = normalized.get(
        "distances",
        []
    )


    extracted = []


    for index, document in enumerate(
        documents
    ):

        if not document:

            continue


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        if (
            index < len(metadatas)
            and isinstance(
                metadatas[index],
                dict
            )
        ):

            metadata = metadatas[index]

        else:

            metadata = {}


        # ----------------------------------------------------
        # Distance
        # ----------------------------------------------------

        if index < len(distances):

            distance = distances[index]

        else:

            distance = None


        extracted.append(
            {
                "text": str(
                    document
                ),

                "source": metadata.get(
                    "source",
                    "Unknown"
                ),

                "page": metadata.get(
                    "page",
                    "Unknown"
                ),

                "chunk_id": metadata.get(
                    "chunk_id",
                    "Unknown"
                ),

                "distance": distance
            }
        )


    return extracted


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results
):
    """
    Convert retrieved documents into LLM context.
    """

    documents = extract_documents(
        results
    )


    if not documents:

        return ""


    context_parts = []


    for index, document in enumerate(
        documents
    ):

        source = document.get(
            "source",
            "Unknown"
        )

        page = document.get(
            "page",
            "Unknown"
        )

        text = document.get(
            "text",
            ""
        )


        context_parts.append(
            f"""
DOCUMENT {index + 1}
Source: {source}
Page: {page}

{text}
"""
        )


    return "\n".join(
        context_parts
    )


# ============================================================
# EXTRACT SOURCES
# ============================================================

def extract_sources(results):
    """
    Extract useful source information for the frontend.

    Each source contains:
        - source
        - page
        - text
        - chunk_id
        - distance
    """

    documents = extract_documents(results)

    sources = []

    seen = set()

    for document in documents:

        source = document.get(
            "source",
            "Unknown"
        )

        page = document.get(
            "page",
            "Unknown"
        )

        text = document.get(
            "text",
            ""
        )

        chunk_id = document.get(
            "chunk_id",
            "Unknown"
        )

        distance = document.get(
            "distance"
        )

        # Avoid duplicate source + page combinations
        key = (
            str(source),
            str(page)
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "source": source,
                "page": page,
                "text": text,
                "chunk_id": chunk_id,
                "distance": distance
            }
        )

    return sources

# ============================================================
# PRINT HEADER
# ============================================================

def print_header():

    print("\n")

    print(
        "=" * 70
    )

    print(
        "SELF-HEALING RAG SYSTEM"
    )

    print(
        "=" * 70
    )


# ============================================================
# PRINT QUERIES
# ============================================================

def print_queries(
    queries
):

    print("\n")

    print(
        "=" * 70
    )

    print(
        "EXPANDED QUERIES"
    )

    print(
        "=" * 70
    )


    for index, query in enumerate(
        queries,
        start=1
    ):

        print(
            f"{index}. {query}"
        )


# ============================================================
# PRINT EVALUATION
# ============================================================

def print_evaluation(
    evaluation
):

    print("\n")

    print(
        "LLM EVALUATION:"
    )

    print(
        evaluation
    )


# ============================================================
# PRINT EVIDENCE
# ============================================================

def print_evidence(
    evidence
):

    print("\n")

    print(
        "EVIDENCE CHECK:"
    )

    print(
        evidence
    )


# ============================================================
# NORMALIZE EVALUATION
# ============================================================

def normalize_evaluation(
    evaluation
):

    if not isinstance(
        evaluation,
        dict
    ):

        return {
            "overall": "BAD",
            "relevance": 0.0,
            "faithfulness": 0.0,
            "evidence_sufficiency": False,
            "query_answer_consistency": False,
            "answer_type": "UNKNOWN",
            "unsupported_claims": [],
            "reasoning": str(
                evaluation
            )
        }


    return {
        "overall": str(
            evaluation.get(
                "overall",
                "BAD"
            )
        ).upper(),

        "relevance": safe_float(
            evaluation.get(
                "relevance",
                0.0
            )
        ),

        "faithfulness": safe_float(
            evaluation.get(
                "faithfulness",
                0.0
            )
        ),

        "evidence_sufficiency":
            safe_bool(
                evaluation.get(
                    "evidence_sufficiency",
                    False
                )
            ),

        "query_answer_consistency":
            safe_bool(
                evaluation.get(
                    "query_answer_consistency",
                    False
                )
            ),

        "answer_type": str(
            evaluation.get(
                "answer_type",
                "UNKNOWN"
            )
        ).upper(),

        "unsupported_claims":
            evaluation.get(
                "unsupported_claims",
                []
            ),

        "reasoning": evaluation.get(
            "reasoning",
            ""
        )
    }


# ============================================================
# NORMALIZE EVIDENCE
# ============================================================

def normalize_evidence(
    evidence
):

    if not isinstance(
        evidence,
        dict
    ):

        return {
            "support_score": 0.0,
            "supported_claims": [],
            "unsupported_claims": [],
            "evidence_sufficient": False
        }


    return {
        "support_score": safe_float(
            evidence.get(
                "support_score",
                0.0
            )
        ),

        "supported_claims":
            evidence.get(
                "supported_claims",
                []
            ),

        "unsupported_claims":
            evidence.get(
                "unsupported_claims",
                []
            ),

        "evidence_sufficient":
            safe_bool(
                evidence.get(
                    "evidence_sufficient",
                    False
                )
            )
    }


# ============================================================
# DETERMINE SUCCESS
# ============================================================

def answer_is_successful(
    evaluation,
    evidence
):
    """
    Decide whether answer can be accepted.
    """

    evaluation = normalize_evaluation(
        evaluation
    )

    evidence = normalize_evidence(
        evidence
    )


    overall = str(
        evaluation.get(
            "overall",
            ""
        )
    ).upper()


    answer_type = str(
        evaluation.get(
            "answer_type",
            ""
        )
    ).upper()


    relevance = safe_float(
        evaluation.get(
            "relevance",
            0
        )
    )


    faithfulness = safe_float(
        evaluation.get(
            "faithfulness",
            0
        )
    )


    consistency = safe_bool(
        evaluation.get(
            "query_answer_consistency",
            False
        )
    )


    evidence_sufficient = safe_bool(
        evidence.get(
            "evidence_sufficient",
            False
        )
    )


    support_score = safe_float(
        evidence.get(
            "support_score",
            0
        )
    )


    # --------------------------------------------------------
    # Strong success
    # --------------------------------------------------------

    if (
        overall == "GOOD"
        and relevance >= 0.5
        and faithfulness >= 0.5
        and consistency
        and evidence_sufficient
        and support_score >= 0.5
    ):

        return True


    # --------------------------------------------------------
    # Supported answer
    # --------------------------------------------------------

    if (
        answer_type == "SUPPORTED"
        and relevance >= 0.5
        and faithfulness >= 0.5
        and consistency
        and evidence_sufficient
        and support_score >= 0.5
    ):

        return True


    return False


# ============================================================
# GENERATE GROUNDED ANSWER
# ============================================================

def generate_grounded_answer(
    question,
    context,
    strong_prompt=False
):
    """
    Generate an answer using the project's generator.
    """

    if not context.strip():

        return (
            "The retrieved documents do not contain "
            "sufficient information to answer this question."
        )


    if generate_answer is None:

        raise ImportError(
            "\nCould not import generate_answer.\n"
            "Expected generate_answer() inside:\n"
            "src/retrieval/baseline_rag.py\n"
        )


    # --------------------------------------------------------
    # Strong prompt
    # --------------------------------------------------------

    if strong_prompt:

        try:

            return generate_answer(
                question,
                context,
                strong_prompt=True
            )

        except TypeError:

            return generate_answer(
                question,
                context
            )


    # --------------------------------------------------------
    # Normal generation
    # --------------------------------------------------------

    return generate_answer(
        question,
        context
    )


# ============================================================
# NORMALIZE EXPANDED QUERIES
# ============================================================

def normalize_queries(
    queries,
    original_question
):
    """
    Make query expansion output predictable.

    Handles:

        ["q1", "q2", "q3"]

    and:

        {"queries": ["q1", "q2"]}

    and accidental nested structures.
    """

    queries = make_json_safe(
        queries
    )


    # --------------------------------------------------------
    # Dictionary format
    # --------------------------------------------------------

    if isinstance(
        queries,
        dict
    ):

        queries = queries.get(
            "queries",
            queries.get(
                "expanded_queries",
                []
            )
        )


    # --------------------------------------------------------
    # Invalid format
    # --------------------------------------------------------

    if not isinstance(
        queries,
        list
    ):

        queries = []


    clean_queries = []


    for query in queries:

        # ----------------------------------------------
        # Nested dictionary
        # ----------------------------------------------

        if isinstance(
            query,
            dict
        ):

            query = query.get(
                "query",
                query.get(
                    "text",
                    ""
                )
            )


        # ----------------------------------------------
        # Nested list
        # ----------------------------------------------

        if isinstance(
            query,
            list
        ):

            for nested_query in query:

                if isinstance(
                    nested_query,
                    str
                ):

                    nested_query = nested_query.strip()

                    if nested_query:

                        clean_queries.append(
                            nested_query
                        )

            continue


        if not isinstance(
            query,
            str
        ):

            continue


        query = query.strip()


        if query:

            clean_queries.append(
                query
            )


    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    final_queries = []

    seen = set()


    for query in (
        [original_question]
        + clean_queries
    ):

        normalized = query.strip()


        if not normalized:

            continue


        key = normalized.lower()


        if key in seen:

            continue


        seen.add(
            key
        )


        final_queries.append(
            normalized
        )


    return final_queries


# ============================================================
# RETRIEVE AND MERGE RESULTS
# ============================================================

def retrieve_documents(
    queries,
    top_k
):
    """
    Retrieve documents for all expanded queries.

    This function is deliberately isolated from the main
    self-healing loop so search-result format problems cannot
    corrupt the rest of the pipeline.
    """

    combined_documents = []

    combined_metadatas = []

    combined_distances = []

    seen_text = set()


    for query in queries:

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        try:

            retrieved = search(
                query,
                top_k=top_k
            )

        except TypeError:

            retrieved = search(
                query,
                top_k
            )


        # ----------------------------------------------------
        # Normalize result
        # ----------------------------------------------------

        normalized = normalize_search_result(
            retrieved
        )


        documents = normalized.get(
            "documents",
            []
        )

        metadatas = normalized.get(
            "metadatas",
            []
        )

        distances = normalized.get(
            "distances",
            []
        )


        # ----------------------------------------------------
        # Add results
        # ----------------------------------------------------

        for index, document in enumerate(
            documents
        ):

            if not document:

                continue


            text = str(
                document
            ).strip()


            if not text:

                continue


            # ----------------------------------------------
            # Deduplicate
            # ----------------------------------------------

            if text in seen_text:

                continue


            seen_text.add(
                text
            )


            # ----------------------------------------------
            # Metadata
            # ----------------------------------------------

            if (
                index < len(metadatas)
                and isinstance(
                    metadatas[index],
                    dict
                )
            ):

                metadata = metadatas[index]

            else:

                metadata = {}


            # ----------------------------------------------
            # Distance
            # ----------------------------------------------

            if index < len(distances):

                distance = distances[index]

            else:

                distance = None


            combined_documents.append(
                text
            )

            combined_metadatas.append(
                metadata
            )

            combined_distances.append(
                distance
            )


    # ========================================================
    # LIMIT RESULTS
    # ========================================================

    combined_documents = (
        combined_documents[
            :top_k
        ]
    )

    combined_metadatas = (
        combined_metadatas[
            :top_k
        ]
    )

    combined_distances = (
        combined_distances[
            :top_k
        ]
    )


    return {
        "documents": [
            combined_documents
        ],

        "metadatas": [
            combined_metadatas
        ],

        "distances": [
            combined_distances
        ]
    }


# ============================================================
# MAIN SELF-HEALING RAG FUNCTION
# ============================================================

def run_self_healing_rag(
    question,
    verbose=True
):
    """
    Run the complete Self-Healing RAG pipeline.

    Returns a structured dictionary suitable for FastAPI
    and the React frontend.
    """

    # ========================================================
    # VALIDATE QUESTION
    # ========================================================

    if not isinstance(
        question,
        str
    ):

        raise ValueError(
            "Question must be a string."
        )


    question = question.strip()


    if not question:

        raise ValueError(
            "Question cannot be empty."
        )


    # ========================================================
    # RESULT OBJECT
    # ========================================================

    result = {

        "question": question,

        "answer": "",

        "risk": {
            "level": "UNKNOWN",
            "score": 0.0,
            "reasoning": ""
        },

        "retrieval": {
            "initial_top_k": DEFAULT_TOP_K,
            "current_top_k": DEFAULT_TOP_K,
            "strategy": "",
            "queries": [],
            "retrieved_documents": 0
        },

        "evaluation": {
            "overall": "BAD",
            "relevance": 0.0,
            "faithfulness": 0.0,
            "evidence_sufficiency": False,
            "query_answer_consistency": False,
            "answer_type": "UNKNOWN",
            "unsupported_claims": [],
            "reasoning": ""
        },

        "evidence": {
            "support_score": 0.0,
            "supported_claims": [],
            "unsupported_claims": [],
            "evidence_sufficient": False
        },

        "sources": [],

        "attempts": 0,

        "status": "FAILED",

        "repair_history": []
    }


    # ========================================================
    # RISK DETECTION
    # ========================================================

    if verbose:

        print("\n")
        print("=" * 70)
        print("RISK DETECTION")
        print("=" * 70)


    risk = detect_risk(
        question
    )


    risk = make_json_safe(
        risk
    )


    if not isinstance(
        risk,
        dict
    ):

        risk = {
            "risk_level": "LOW",
            "risk_score": 0.0,
            "reasoning": str(
                risk
            )
        }


    result["risk"] = {

        "level": risk.get(
            "risk_level",
            "UNKNOWN"
        ),

        "score": safe_float(
            risk.get(
                "risk_score",
                0
            )
        ),

        "reasoning": risk.get(
            "reasoning",
            ""
        )
    }


    if verbose:

        print(
            f"Risk Level : "
            f"{result['risk']['level']}"
        )

        print(
            f"Risk Score : "
            f"{result['risk']['score']}"
        )

        print(
            f"Reasoning  : "
            f"{result['risk']['reasoning']}"
        )


    # ========================================================
    # ADAPTIVE RETRIEVAL
    # ========================================================

    retrieval_strategy = choose_retrieval_strategy(
        risk
    )


    retrieval_strategy = make_json_safe(
        retrieval_strategy
    )


    if isinstance(
        retrieval_strategy,
        dict
    ):

        top_k = safe_float(
            retrieval_strategy.get(
                "top_k",
                DEFAULT_TOP_K
            ),
            DEFAULT_TOP_K
        )

        strategy_name = retrieval_strategy.get(
            "strategy",
            "Adaptive retrieval"
        )

    else:

        top_k = DEFAULT_TOP_K

        strategy_name = str(
            retrieval_strategy
        )


    top_k = int(
        max(
            1,
            min(
                MAX_TOP_K,
                top_k
            )
        )
    )


    result["retrieval"][
        "initial_top_k"
    ] = top_k


    result["retrieval"][
        "current_top_k"
    ] = top_k


    result["retrieval"][
        "strategy"
    ] = strategy_name


    if verbose:

        print("\n")
        print("=" * 70)
        print("ADAPTIVE RETRIEVAL")
        print("=" * 70)

        print(
            f"Risk Level : "
            f"{result['risk']['level']}"
        )

        print(
            f"Initial TOP-K : "
            f"{top_k}"
        )

        print(
            f"Strategy : "
            f"{strategy_name}"
        )


    # ========================================================
    # SELF-HEALING STATE
    # ========================================================

    current_top_k = top_k

    strong_prompt = False

    final_answer = ""

    final_results = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]]
    }


    # ========================================================
    # SELF-HEALING LOOP
    # ========================================================

    for attempt in range(
        1,
        MAX_REPAIR_ATTEMPTS + 1
    ):

        result["attempts"] = attempt

        result["retrieval"][
            "current_top_k"
        ] = current_top_k


        # ====================================================
        # ATTEMPT HEADER
        # ====================================================

        if verbose:

            print("\n")
            print("=" * 70)

            print(
                f"ATTEMPT {attempt}"
            )

            print(
                f"TOP-K = {current_top_k}"
            )

            print("=" * 70)


        # ====================================================
        # RETRIEVAL
        # ====================================================

        if verbose:

            print(
                "\nRetrieving documents..."
            )


        # ----------------------------------------------------
        # Query expansion
        # ----------------------------------------------------

        expanded_queries = expand_query(
            question
        )


        queries = normalize_queries(
            expanded_queries,
            question
        )


        result["retrieval"][
            "queries"
        ] = queries


        if verbose:

            print_queries(
                queries
            )


        # ----------------------------------------------------
        # Retrieve all queries
        # ----------------------------------------------------

        final_results = retrieve_documents(
            queries,
            current_top_k
        )


        # ----------------------------------------------------
        # Count documents
        # ----------------------------------------------------

        retrieved_documents = extract_documents(
            final_results
        )


        result["retrieval"][
            "retrieved_documents"
        ] = len(
            retrieved_documents
        )


        if verbose:

            print(
                f"\nRetrieved "
                f"{len(retrieved_documents)} "
                f"unique chunks."
            )


        # ====================================================
        # CONTEXT
        # ====================================================

        context = build_context(
            final_results
        )


        # ====================================================
        # SOURCES
        # ====================================================

        result["sources"] = extract_sources(
            final_results
        )


        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        if verbose:

            print(
                "\nGenerating answer..."
            )


        final_answer = generate_grounded_answer(
            question,
            context,
            strong_prompt=strong_prompt
        )


        if final_answer is None:

            final_answer = ""


        final_answer = str(
            final_answer
        ).strip()


        if verbose:

            print(
                "\nANSWER:"
            )

            print(
                final_answer
            )


        # ====================================================
        # LLM EVALUATION
        # ====================================================

        if verbose:

            print(
                "\nEvaluating answer..."
            )


        try:

            evaluation = evaluate_answer(
                question,
                final_answer,
                context
            )

        except TypeError:

            try:

                evaluation = evaluate_answer(
                    question,
                    final_answer,
                    final_results
                )

            except TypeError:

                evaluation = evaluate_answer(
                    question,
                    final_answer
                )


        evaluation = make_json_safe(
            evaluation
        )


        evaluation = normalize_evaluation(
            evaluation
        )


        result["evaluation"] = evaluation


        if verbose:

            print_evaluation(
                evaluation
            )


        # ====================================================
        # EVIDENCE CHECK
        # ====================================================

        if verbose:

            print(
                "\nChecking evidence..."
            )


        try:

            evidence = check_evidence(
                question,
                final_answer,
                context
            )

        except TypeError:

            try:

                evidence = check_evidence(
                    final_answer,
                    context
                )

            except TypeError:

                evidence = check_evidence(
                    final_answer
                )


        evidence = make_json_safe(
            evidence
        )


        evidence = normalize_evidence(
            evidence
        )


        result["evidence"] = evidence


        if verbose:

            print_evidence(
                evidence
            )


        # ====================================================
        # FINAL EVALUATION
        # ====================================================

        successful = answer_is_successful(
            evaluation,
            evidence
        )


        if verbose:

            print("\n")
            print("=" * 70)
            print("FINAL EVALUATION")
            print("=" * 70)

            print(
                f"LLM overall: "
                f"{evaluation['overall']}"
            )

            print(
                f"LLM relevance: "
                f"{evaluation['relevance']}"
            )

            print(
                f"LLM faithfulness: "
                f"{evaluation['faithfulness']}"
            )

            print(
                f"Query-answer consistency: "
                f"{evaluation['query_answer_consistency']}"
            )

            print(
                f"Evidence support score: "
                f"{evidence['support_score']}"
            )

            print(
                "Evidence: "
                + (
                    "SUFFICIENT"
                    if evidence[
                        "evidence_sufficient"
                    ]
                    else "INSUFFICIENT"
                )
            )


        # ====================================================
        # SUCCESS
        # ====================================================

        if successful:

            result["status"] = "SUCCESS"


            if verbose:

                print("\n")
                print("=" * 70)
                print("SUCCESS")
                print("=" * 70)

                print(
                    f"Answer accepted on "
                    f"attempt {attempt}."
                )


            break


        # ====================================================
        # LAST ATTEMPT
        # ====================================================

        if attempt >= MAX_REPAIR_ATTEMPTS:

            if verbose:

                print("\n")
                print("=" * 70)
                print(
                    "MAXIMUM REPAIR ATTEMPTS REACHED"
                )
                print("=" * 70)

                print(
                    "The system could not produce "
                    "a sufficiently supported answer."
                )


            break


        # ====================================================
        # FAILURE DIAGNOSIS
        # ====================================================

        if verbose:

            print(
                "\nDiagnosing failure..."
            )


        try:

            diagnosis = diagnose_failure(
                question,
                final_answer,
                evaluation,
                evidence
            )

        except TypeError:

            try:

                diagnosis = diagnose_failure(
                    question,
                    final_answer,
                    evaluation
                )

            except TypeError:

                diagnosis = diagnose_failure(
                    evaluation,
                    evidence
                )


        diagnosis = make_json_safe(
            diagnosis
        )


        if verbose:

            print(
                "\nDIAGNOSIS:"
            )

            print(
                diagnosis
            )


        # ====================================================
        # DETERMINE REPAIR ACTION
        # ====================================================

        repair_action = ""


        if isinstance(
            diagnosis,
            dict
        ):

            repair_action = diagnosis.get(
                "repair_action",
                ""
            )


        if verbose:

            print(
                "\nREPAIR STRATEGY:"
            )


        # ====================================================
        # REPAIR RETRIEVAL
        # ====================================================

        try:

            repair_result = repair_retrieval(
                current_top_k,
                repair_action
            )

        except TypeError:

            try:

                repair_result = repair_retrieval(
                    current_top_k
                )

            except TypeError:

                repair_result = None


        repair_result = make_json_safe(
            repair_result
        )


        # ====================================================
        # NORMALIZE REPAIR RESULT
        # ====================================================

        new_top_k = None

        action = repair_action


        if isinstance(
            repair_result,
            dict
        ):

            action = repair_result.get(
                "action",
                action
            )

            new_top_k = repair_result.get(
                "new_top_k",
                None
            )


        elif isinstance(
            repair_result,
            int
        ):

            new_top_k = repair_result


        # ====================================================
        # DEFAULT SELF-HEALING PROGRESSION
        #
        # 3 → 6 → 12 → 20
        # ====================================================

        if new_top_k is None:

            if current_top_k < 6:

                new_top_k = 6

            elif current_top_k < 12:

                new_top_k = 12

            elif current_top_k < 20:

                new_top_k = 20

            else:

                new_top_k = current_top_k


        new_top_k = int(
            max(
                current_top_k,
                min(
                    MAX_TOP_K,
                    int(new_top_k)
                )
            )
        )


        # ====================================================
        # GENERATION REPAIR
        # ====================================================

        if action in (
            "strengthen_prompt",
            "generation_repair",
            "repair_generation"
        ):

            strong_prompt = True


        # ====================================================
        # RETRIEVAL REPAIR
        # ====================================================

        retrieval_changed = (
            new_top_k
            != current_top_k
        )


        if retrieval_changed:

            strong_prompt = False


        # ====================================================
        # REPAIR HISTORY
        # ====================================================

        if isinstance(
            diagnosis,
            dict
        ):

            failure_type = diagnosis.get(
                "failure_type",
                "UNKNOWN"
            )

        else:

            failure_type = "UNKNOWN"


        repair_record = {

            "attempt": attempt,

            "failure_type":
                failure_type,

            "repair_action":
                action,

            "old_top_k":
                current_top_k,

            "new_top_k":
                new_top_k
        }


        result[
            "repair_history"
        ].append(
            repair_record
        )


        if verbose:

            print(
                repair_record
            )

            print(
                "\nApplying repair..."
            )

            print(
                f"Repair action: "
                f"{action}"
            )

            print(
                f"Old TOP-K: "
                f"{current_top_k}"
            )

            print(
                f"New TOP-K: "
                f"{new_top_k}"
            )


        current_top_k = new_top_k


        if verbose:

            if strong_prompt:

                print(
                    "\nGeneration repair activated."
                )

                print(
                    "Using stronger "
                    "evidence-grounding prompt."
                )

            elif retrieval_changed:

                print(
                    "\nRetrieval repair activated."
                )

                print(
                    f"Next attempt will retrieve "
                    f"{current_top_k} chunks."
                )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    result["answer"] = final_answer


    result["retrieval"][
        "current_top_k"
    ] = current_top_k


    # ========================================================
    # FINAL ANSWER
    # ========================================================

    if verbose:

        print("\n")
        print("=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)

        print(
            final_answer
        )

        print("\n")
        print("=" * 70)
        print(
            "SELF-HEALING RAG COMPLETE"
        )
        print("=" * 70)


    return make_json_safe(
        result
    )


# ============================================================
# TERMINAL MODE
# ============================================================

def main():

    print_header()


    print(
        "\nEnter your question:"
    )


    question = input(
        "> "
    ).strip()


    if not question:

        print(
            "\nPlease enter a question."
        )

        return


    try:

        run_self_healing_rag(
            question,
            verbose=True
        )


    except Exception as e:

        print("\n")
        print("=" * 70)
        print("SYSTEM ERROR")
        print("=" * 70)

        print(
            str(e)
        )

        print(
            "\nTraceback:"
        )

        traceback.print_exc()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()