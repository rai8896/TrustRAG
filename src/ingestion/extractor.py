from pathlib import Path
from pypdf import PdfReader
import re


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "documents"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    # Replace multiple spaces/tabs
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text
    )

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
# EXTRACT ALL DOCUMENTS
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
# TEST
# ============================================================

def main():

    pages = extract_all_documents()

    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)

    print(
        f"Total extracted pages: {len(pages)}"
    )

    for page in pages[:3]:

        print("\n" + "-" * 70)

        print(
            f"Source : {page['source']}"
        )

        print(
            f"Page   : {page['page']}"
        )

        print("-" * 70)

        print(
            page["text"][:1000]
        )


if __name__ == "__main__":
    main()