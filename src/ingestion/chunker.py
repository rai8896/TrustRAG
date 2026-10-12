from pathlib import Path
import re


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/documents")

# Target chunk size
CHUNK_SIZE = 700

# Overlap between neighboring chunks
CHUNK_OVERLAP = 100


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean extracted PDF text while preserving meaningful content.
    """

    if not text:
        return ""

    # Normalize spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove unnecessary spaces at beginning/end of lines
    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:
            lines.append(line)

    text = "\n".join(lines)

    return text.strip()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pages_from_pdf(pdf_path):
    """
    Extract text from every PDF page separately.

    Returns:

    [
        {
            "source": "example.pdf",
            "page": 1,
            "text": "..."
        }
    ]
    """

    from pypdf import PdfReader

    pages = []

    try:

        reader = PdfReader(pdf_path)

    except Exception as e:

        print(
            f"ERROR reading {pdf_path.name}: {e}"
        )

        return pages


    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            raw_text = page.extract_text()

            text = clean_text(raw_text)

            if not text:
                continue

            pages.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": text
                }
            )

        except Exception as e:

            print(
                f"WARNING: Could not extract "
                f"page {page_number} "
                f"from {pdf_path.name}: {e}"
            )


    return pages


# ============================================================
# EXTRACT ALL PDF DOCUMENTS
# ============================================================

def extract_all_documents():

    all_pages = []

    pdf_files = sorted(
        DATA_DIR.glob("*.pdf")
    )

    print(
        f"\nFound {len(pdf_files)} PDF file(s)."
    )

    for pdf_path in pdf_files:

        print("\n" + "=" * 70)

        print(
            f"PROCESSING: {pdf_path.name}"
        )

        print("=" * 70)

        pages = extract_pages_from_pdf(
            pdf_path
        )

        print(
            f"Extracted {len(pages)} "
            f"pages containing text."
        )

        all_pages.extend(pages)

    return all_pages


# ============================================================
# SPLIT TEXT INTO SENTENCES
# ============================================================

def split_into_sentences(text):
    """
    Split text into reasonably meaningful sentences.

    This is intentionally simple because government PDFs
    often contain numbered clauses and bullet points.
    """

    # Protect common abbreviations
    text = text.replace(
        "e.g.",
        "e.g"
    )

    text = text.replace(
        "i.e.",
        "i.e"
    )

    # Split after sentence-ending punctuation
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# CREATE SEMANTIC CHUNKS
# ============================================================

def create_semantic_chunks(
    pages,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
):
    """
    Create semantic chunks from extracted PDF pages.

    Important properties:

    - Chunks do not cross PDF pages.
    - Sentences are kept together where possible.
    - Metadata such as source and page is preserved.
    - A small overlap is added between neighboring chunks.
    """

    chunks = []

    global_chunk_id = 0


    for page in pages:

        source = page["source"]

        page_number = page["page"]

        text = page["text"].strip()

        if not text:
            continue


        # ----------------------------------------------------
        # Split page into sentences
        # ----------------------------------------------------

        sentences = split_into_sentences(
            text
        )

        if not sentences:
            continue


        current_chunk = []

        current_length = 0


        for sentence in sentences:

            sentence_length = len(
                sentence
            )


            # ------------------------------------------------
            # If sentence itself is larger than chunk size
            # ------------------------------------------------

            if sentence_length > chunk_size:

                # Flush current chunk first

                if current_chunk:

                    chunk_text = " ".join(
                        current_chunk
                    )

                    chunks.append(
                        {
                            "chunk_id": global_chunk_id,
                            "source": source,
                            "page": page_number,
                            "text": chunk_text
                        }
                    )

                    global_chunk_id += 1

                    current_chunk = []

                    current_length = 0


                # Split oversized sentence
                # into smaller word-based pieces

                words = sentence.split()

                piece = []
                piece_length = 0

                for word in words:

                    additional_length = (
                        len(word) + 1
                    )

                    if (
                        piece
                        and
                        piece_length
                        + additional_length
                        > chunk_size
                    ):

                        piece_text = " ".join(
                            piece
                        )

                        chunks.append(
                            {
                                "chunk_id": global_chunk_id,
                                "source": source,
                                "page": page_number,
                                "text": piece_text
                            }
                        )

                        global_chunk_id += 1

                        piece = []

                        piece_length = 0


                    piece.append(word)

                    piece_length += (
                        additional_length
                    )


                if piece:

                    piece_text = " ".join(
                        piece
                    )

                    chunks.append(
                        {
                            "chunk_id": global_chunk_id,
                            "source": source,
                            "page": page_number,
                            "text": piece_text
                        }
                    )

                    global_chunk_id += 1

                continue


            # ------------------------------------------------
            # Check whether sentence fits
            # ------------------------------------------------

            if (
                current_chunk
                and
                current_length
                + sentence_length
                + 1
                > chunk_size
            ):

                # Create chunk

                chunk_text = " ".join(
                    current_chunk
                )

                chunks.append(
                    {
                        "chunk_id": global_chunk_id,
                        "source": source,
                        "page": page_number,
                        "text": chunk_text
                    }
                )

                global_chunk_id += 1


                # --------------------------------------------
                # Create overlap
                # --------------------------------------------

                overlap_sentences = []

                overlap_length = 0

                for previous_sentence in reversed(
                    current_chunk
                ):

                    if (
                        overlap_length
                        + len(previous_sentence)
                        + 1
                        <= chunk_overlap
                    ):

                        overlap_sentences.insert(
                            0,
                            previous_sentence
                        )

                        overlap_length += (
                            len(previous_sentence)
                            + 1
                        )

                    else:

                        break


                current_chunk = (
                    overlap_sentences
                )

                current_length = (
                    overlap_length
                )


            # ------------------------------------------------
            # Add current sentence
            # ------------------------------------------------

            current_chunk.append(
                sentence
            )

            current_length += (
                sentence_length + 1
            )


        # ----------------------------------------------------
        # Flush final chunk of page
        # ----------------------------------------------------

        if current_chunk:

            chunk_text = " ".join(
                current_chunk
            )

            chunks.append(
                {
                    "chunk_id": global_chunk_id,
                    "source": source,
                    "page": page_number,
                    "text": chunk_text
                }
            )

            global_chunk_id += 1


    return chunks


# ============================================================
# DISPLAY SAMPLE CHUNKS
# ============================================================

def display_sample_chunks(
    chunks,
    number=10
):

    print("\n" + "=" * 70)
    print("SAMPLE CHUNKS")
    print("=" * 70)


    for chunk in chunks[:number]:

        print(
            "\n" + "-" * 70
        )

        print(
            f"Chunk ID : "
            f"{chunk['chunk_id']}"
        )

        print(
            f"Source   : "
            f"{chunk['source']}"
        )

        print(
            f"Page     : "
            f"{chunk['page']}"
        )

        print(
            f"Length   : "
            f"{len(chunk['text'])}"
        )

        print("-" * 70)

        print(
            chunk["text"][:700]
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Extract PDF pages
    # --------------------------------------------------------

    from src.ingestion.ingest import (
        extract_all_documents
    )

    print(
        "\nExtracting PDF pages..."
    )

    pages = extract_all_documents()

    print(
        f"\nExtracted pages: "
        f"{len(pages)}"
    )


    # --------------------------------------------------------
    # Create semantic chunks
    # --------------------------------------------------------

    print(
        "\nCreating semantic chunks..."
    )

    chunks = create_semantic_chunks(
        pages
    )

    print(
        f"Created chunks: "
        f"{len(chunks)}"
    )


    # --------------------------------------------------------
    # Display samples
    # --------------------------------------------------------

    display_sample_chunks(
        chunks,
        number=10
    )


    # --------------------------------------------------------
    # Create vector store
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CREATING VECTOR STORE"
    )

    print(
        "=" * 70
    )


    from src.retrieval.vector_store import (
        create_vector_store
    )

    create_vector_store(
        chunks
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "VECTOR STORE CREATION COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()