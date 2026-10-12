import shutil
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = "chroma_db"
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
    path=CHROMA_DIR
)


# ============================================================
# DELETE OLD COLLECTION
# ============================================================

def reset_collection():
    """
    Delete the old collection and create a fresh one.

    This is important because the old ChromaDB may contain
    the previous page-level chunks.
    """

    print("\nResetting ChromaDB collection...")

    try:

        client.delete_collection(
            name=COLLECTION_NAME
        )

        print("Old collection deleted.")

    except Exception:

        print("Old collection did not exist.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "PMAY-U 2.0 government policy documents"
        }
    )

    print("New collection created.")

    return collection


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store(chunks):
    """
    Convert chunks into embeddings and store them in ChromaDB.

    Expected chunk format:

    {
        "chunk_id": 0,
        "text": "...",
        "source": "...",
        "page": 17
    }
    """

    if not chunks:

        raise ValueError(
            "No chunks were provided."
        )

    collection = reset_collection()

    print(
        f"\nCreating embeddings for {len(chunks)} chunks..."
    )

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for index, chunk in enumerate(chunks):

        text = chunk.get("text", "").strip()

        if not text:
            continue

        # ----------------------------------------------------
        # Create embedding
        # ----------------------------------------------------

        embedding = embedding_model.encode(
            text,
            normalize_embeddings=True
        )

        # ----------------------------------------------------
        # Store document
        # ----------------------------------------------------

        documents.append(text)

        embeddings.append(
            embedding.tolist()
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadatas.append(
            {
                "source": str(
                    chunk.get(
                        "source",
                        "unknown"
                    )
                ),

                "page": int(
                    chunk.get(
                        "page",
                        chunk.get(
                            "page_number",
                            0
                        )
                    )
                ),

                "chunk_id": str(
                    chunk.get(
                        "chunk_id",
                        index
                    )
                )
            }
        )

        # ----------------------------------------------------
        # Unique Chroma ID
        # ----------------------------------------------------

        ids.append(
            f"chunk_{index}"
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (index + 1) % 50 == 0:

            print(
                f"Embedded {index + 1}/"
                f"{len(chunks)} chunks..."
            )

    # ========================================================
    # ADD TO CHROMADB
    # ========================================================

    print("\nAdding embeddings to ChromaDB...")

    # ChromaDB can handle batches, but using one batch here
    # keeps the implementation simple for this project.

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"Successfully stored {len(documents)} chunks."
    )

    return collection


# ============================================================
# LOAD EXISTING COLLECTION
# ============================================================

def get_collection():
    """
    Load the existing ChromaDB collection.
    """

    try:

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

        return collection

    except Exception as e:

        raise RuntimeError(
            f"Could not load ChromaDB collection "
            f"'{COLLECTION_NAME}'. "
            f"Run the ingestion/vector-store creation "
            f"first.\n\nOriginal error: {e}"
        )


# ============================================================
# SEARCH VECTOR STORE
# ============================================================

def search_collection(
    query,
    top_k=5
):
    """
    Search the vector database using semantic similarity.
    """

    collection = get_collection()

    # Convert query to embedding

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

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

    return results


# ============================================================
# DISPLAY SEARCH RESULTS
# ============================================================

def display_results(
    results,
    query
):

    print("\n")
    print("=" * 70)
    print("VECTOR SEARCH RESULTS")
    print("=" * 70)

    print(
        f"\nQuery: {query}"
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    if not documents:

        print(
            "\nNo results found."
        )

        return

    for i, document in enumerate(
        documents
    ):

        metadata = (
            metadatas[i]
            if i < len(metadatas)
            else {}
        )

        distance = (
            distances[i]
            if i < len(distances)
            else None
        )

        print("\n" + "-" * 70)

        print(
            f"RESULT {i + 1}"
        )

        print(
            f"Source : "
            f"{metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page   : "
            f"{metadata.get('page', 'Unknown')}"
        )

        print(
            f"Chunk  : "
            f"{metadata.get('chunk_id', 'Unknown')}"
        )

        if distance is not None:

            print(
                f"Distance : "
                f"{distance:.4f}"
            )

        print("\nText:")

        print(document)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("VECTOR STORE TEST")
    print("=" * 70)

    collection = get_collection()

    count = collection.count()

    print(
        f"\nChunks currently stored: {count}"
    )

    if count == 0:

        print(
            "\nWARNING: ChromaDB collection is empty."
        )

        return

    # --------------------------------------------------------
    # Test query
    # --------------------------------------------------------

    query = (
        "What are the income limits for "
        "EWS, LIG and MIG under PMAY-U 2.0?"
    )

    results = search_collection(
        query,
        top_k=5
    )

    display_results(
        results,
        query
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()