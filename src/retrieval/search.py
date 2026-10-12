from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATH
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "government_policies"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("Embedding model loaded.")


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# ============================================================
# GET COLLECTION
# ============================================================

def get_collection():

    try:

        return client.get_collection(
            name=COLLECTION_NAME
        )

    except Exception as e:

        raise RuntimeError(
            f"\nChromaDB collection "
            f"'{COLLECTION_NAME}' does not exist.\n"
            f"Please run ingestion first.\n\n"
            f"Original error: {e}"
        )


# ============================================================
# SEARCH
# ============================================================

def search(
    query,
    top_k=5
):
    """
    Perform semantic search.

    IMPORTANT:
    This function returns a DICTIONARY,
    not a list.

    Expected structure:

    {
        "documents": [...],
        "metadatas": [...],
        "distances": [...]
    }
    """

    collection = get_collection()

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # Query ChromaDB
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    # --------------------------------------------------------
    # Normalize ChromaDB result
    # --------------------------------------------------------

    documents = results.get(
        "documents",
        [[]]
    )

    metadatas = results.get(
        "metadatas",
        [[]]
    )

    distances = results.get(
        "distances",
        [[]]
    )

    # Chroma returns nested lists because there may
    # be multiple queries.

    documents = (
        documents[0]
        if documents
        else []
    )

    metadatas = (
        metadatas[0]
        if metadatas
        else []
    )

    distances = (
        distances[0]
        if distances
        else []
    )

    # --------------------------------------------------------
    # Return normalized dictionary
    # --------------------------------------------------------

    return {
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances
    }


# ============================================================
# TEST
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("SEARCH TEST")
    print("=" * 70)

    query = (
        "What are the four verticals "
        "of PMAY-U 2.0?"
    )

    results = search(
        query,
        top_k=3
    )

    print(
        f"\nDocuments found: "
        f"{len(results['documents'])}"
    )

    for i, document in enumerate(
        results["documents"]
    ):

        metadata = (
            results["metadatas"][i]
            if i < len(results["metadatas"])
            else {}
        )

        distance = (
            results["distances"][i]
            if i < len(results["distances"])
            else None
        )

        print("\n" + "-" * 70)

        print(
            f"RESULT {i + 1}"
        )

        print(
            f"Source: "
            f"{metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page: "
            f"{metadata.get('page', 'Unknown')}"
        )

        if distance is not None:

            print(
                f"Distance: "
                f"{distance:.4f}"
            )

        print("\nText:")

        print(document)


if __name__ == "__main__":

    main()