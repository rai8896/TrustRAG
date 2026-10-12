from pathlib import Path
import sys

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# FIX PROJECT ROOT PATH
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT EXTRACTION + CHUNKING
# ============================================================

from src.ingestion.chunker import create_semantic_chunks
from src.ingestion.extractor import extract_all_documents


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "government_policies"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# ============================================================
# CREATE / GET COLLECTION
# ============================================================

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# BUILD CHROMADB DOCUMENTS
# ============================================================

def build_documents(chunks):

    documents = []
    metadatas = []
    ids = []

    for index, chunk in enumerate(chunks):

        text = chunk["text"]

        source = chunk["source"]

        page = chunk["page"]

        chunk_id = f"{source}_{page}_{index}"

        documents.append(text)

        metadatas.append(
            {
                "source": source,
                "page": int(page),
                "chunk_id": chunk_id
            }
        )

        ids.append(chunk_id)

    return documents, metadatas, ids


# ============================================================
# INGEST CHUNKS
# ============================================================

def ingest_chunks(chunks):

    if not chunks:

        print("\nNo chunks found.")

        return


    print("\n" + "=" * 70)
    print("CREATING EMBEDDINGS")
    print("=" * 70)

    documents, metadatas, ids = build_documents(
        chunks
    )

    print(
        f"Chunks to embed: {len(documents)}"
    )


    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True
    )


    print("\nEmbeddings created.")


    # --------------------------------------------------------
    # Clear old collection
    #
    # This is important while we are developing the project.
    # It prevents old chunks from remaining in ChromaDB.
    # --------------------------------------------------------

    try:

        client.delete_collection(
            name=COLLECTION_NAME
        )

        print(
            "\nOld ChromaDB collection deleted."
        )

    except Exception:

        pass


    # --------------------------------------------------------
    # Create fresh collection
    # --------------------------------------------------------

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )


    # --------------------------------------------------------
    # Add data
    # --------------------------------------------------------

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=[
            embedding.tolist()
            for embedding in embeddings
        ]
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHROMADB INGESTION COMPLETE")
    print("=" * 70)

    print(
        f"Collection : {COLLECTION_NAME}"
    )

    print(
        f"Chunks     : {len(documents)}"
    )

    print(
        f"Embeddings : {len(embeddings)}"
    )

    print(
        f"Database   : {CHROMA_DIR}"
    )

    print(
        "Status     : SUCCESS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("SELF-HEALING RAG - DOCUMENT INGESTION")
    print("=" * 70)


    # --------------------------------------------------------
    # Extract pages
    # --------------------------------------------------------

    print("\nExtracting PDF pages...")

    pages = extract_all_documents()

    print(
        f"\nExtracted pages: {len(pages)}"
    )


    # --------------------------------------------------------
    # Create semantic chunks
    # --------------------------------------------------------

    print("\nCreating semantic chunks...")

    chunks = create_semantic_chunks(
        pages
    )

    print(
        f"Created chunks: {len(chunks)}"
    )


    # --------------------------------------------------------
    # Store in ChromaDB
    # --------------------------------------------------------

    ingest_chunks(chunks)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()