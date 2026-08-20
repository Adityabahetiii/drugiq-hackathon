"""
PDF Processor
-------------
What it does:
  - Reads every PDF from the data/pdfs/ folder
  - Extracts text page by page (so we always know which page text came from)
  - Splits text into overlapping chunks of ~500 words
  - Tags every chunk with metadata: drug name, page number, source file

Why chunks?
  AI models can't read a 100-page PDF all at once.
  We break it into small pieces so we can find just the relevant piece
  when a question is asked, instead of sending the whole PDF every time.

Why overlap?
  If a sentence is cut across two chunks, overlap ensures it appears
  fully in at least one chunk — so no information is lost at the edges.
"""

import os
import re
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_page_text(page) -> str:
    """
    Extracts a pdfplumber page's text, splitting two-column layouts (common
    on drug label cover/highlights pages) into their own left-then-right
    reading order first.

    pdfplumber's default extract_text() groups words into lines purely by
    y-position across the full page width. On a two-column page that
    interleaves unrelated left- and right-column sentences into one jumbled
    line — e.g. "...is a Janus kinase (JAK) inhibitor. release tablets
    should be made by the healthcare provider." mixes the indications
    column with the dosage column mid-sentence. That corrupted text then
    gets chunked and embedded as-is, and later confuses anything that tries
    to locate it verbatim (like the PDF-viewer highlighter).

    Detects a column layout by checking whether a vertical gap near the
    page's horizontal center is never crossed by a word's bounding box; if
    so, extracts the left and right halves separately and concatenates them
    in proper order instead of pdfplumber's row-interleaved default.
    """
    words = page.extract_words()
    if not words:
        return page.extract_text() or ""

    page_width = page.width
    mid = page_width / 2
    tolerance = page_width * 0.03

    crosses_middle = any(w["x0"] < mid - tolerance and w["x1"] > mid + tolerance for w in words)
    left_words = [w for w in words if w["x1"] <= mid + 2]
    right_words = [w for w in words if w["x0"] >= mid - 2]
    covered = len(left_words) + len(right_words)
    is_two_column = (
        not crosses_middle
        and covered >= len(words) * 0.9
        and len(left_words) > 3
        and len(right_words) > 3
    )

    if not is_two_column:
        return page.extract_text() or ""

    left_text = page.within_bbox((0, 0, mid, page.height)).extract_text() or ""
    right_text = page.within_bbox((mid, 0, page_width, page.height)).extract_text() or ""
    return (left_text + "\n" + right_text).strip()


# FDA-label boilerplate that can appear above the actual product name —
# especially a boxed/black-box warning, which older-style labels (e.g.
# generic injectable labels) print before the drug name ever shows up.
# Matched as a line prefix so "WARNING: SERIOUS INFECTIONS..." and a bare
# "WARNINGS" heading are both caught, not just an exact "WARNINGS" line.
SECTION_HEADER_PREFIXES = (
    'WARNING', 'BOXED WARNING', 'BLACK BOX WARNING',
    'HIGHLIGHTS OF PRESCRIBING INFORMATION', 'MEDICATION GUIDE',
    'PRESCRIBING INFORMATION', 'FULL PRESCRIBING INFORMATION',
    'PATIENT INFORMATION', 'PACKAGE INSERT', 'RX ONLY',
    'IMPORTANT SAFETY INFORMATION', 'INDICATIONS AND USAGE',
    'DOSAGE AND ADMINISTRATION', 'CONTRAINDICATIONS', 'PRECAUTIONS',
    'ADVERSE REACTIONS', 'DRUG INTERACTIONS', 'USE IN SPECIFIC POPULATIONS',
    'CLINICAL PHARMACOLOGY', 'HOW SUPPLIED', 'DESCRIPTION',
    'RECENT MAJOR CHANGES', 'TABLE OF CONTENTS', 'THIS LABEL',
    'FOR CURRENT LABELING',
)

# Formulation/pharmacopeial words that tag along after the actual drug name
# on older-style labels (e.g. "Methotrexate Injection, USP") — stripped off
# so the detected name is just the drug, not "Methotrexate Injection Usp".
DOSAGE_FORM_SUFFIXES = {
    'injection', 'tablets', 'tablet', 'capsules', 'capsule', 'solution',
    'oral', 'cream', 'ointment', 'gel', 'suspension', 'syrup', 'powder',
    'vial', 'usp', 'nf', 'inc', 'llc', 'kit', 'lq', 'xr', 'er', 'extended',
    'release', 'concentrate', 'lyophilized',
}


def _is_section_header_line(line: str) -> bool:
    """True for FDA-label boilerplate (a section header, boxed warning,
    disclaimer, ...) so name-detection can skip past it instead of
    mistaking it for the product name."""
    normalized = re.sub(r'[^A-Za-z\s]', ' ', line).upper()
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    if not normalized:
        return True
    return any(normalized.startswith(h) for h in SECTION_HEADER_PREFIXES)


def _looks_like_product_name_line(line: str) -> bool:
    """
    A short line where every word is capitalized (Title Case, or an
    all-caps abbreviation like "USP") — the shape of a product name line,
    as opposed to an ordinary sentence full of lowercase articles and
    prepositions ("This label may not be the latest...").
    """
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", line)
    if not words or len(words) > 6:
        return False
    return all(w[0].isupper() for w in words if len(w) >= 2)


def _strip_dosage_form_suffix(name: str) -> str:
    """Trims trailing formulation words off a detected name line, e.g.
    "Methotrexate Injection, USP" -> "Methotrexate"."""
    words = re.sub(r',', ' ', name).split()
    while len(words) > 1 and words[-1].lower().strip('.') in DOSAGE_FORM_SUFFIXES:
        words.pop()
    return ' '.join(words) if words else name


def extract_drug_name_from_content(filepath: str) -> str:
    """
    Tries to extract the actual drug name from the first page of the PDF.
    Looks for patterns like:
      - "RINVOQ® (upadacitinib)"
      - "HIGHLIGHTS OF PRESCRIBING INFORMATION ... DRUG_NAME"
      - A Title-Case product name line (e.g. "Methotrexate Injection, USP")
      - Brand name in large/bold ALL-CAPS text on first page
    Section headers (WARNINGS, BOXED WARNING, MEDICATION GUIDE, ...) are
    skipped everywhere below — some labels print a boxed warning before the
    drug name ever appears, and without this an all-caps "WARNINGS" heading
    would otherwise get mistaken for the product name.
    Falls back to filename-based extraction if nothing is found.
    """
    try:
        with pdfplumber.open(filepath) as pdf:
            if pdf.pages:
                first_page_text = extract_page_text(pdf.pages[0])

                # Pattern 1: Look for "BRAND_NAME® (generic_name)" or "BRAND_NAME (generic_name)"
                match = re.search(
                    r'([A-Z][A-Z\s]{2,20})[®™]?\s*\(([a-zA-Z\-]+)\)',
                    first_page_text
                )
                if match and not _is_section_header_line(match.group(1)):
                    brand = match.group(1).strip().title()
                    generic = match.group(2).strip().lower()
                    return f"{brand} ({generic})"

                # Pattern 2: Look for "HIGHLIGHTS OF PRESCRIBING INFORMATION"
                # The drug name often appears right after or near this line
                highlight_match = re.search(
                    r'HIGHLIGHTS\s+OF\s+PRESCRIBING\s+INFORMATION.*?\n\s*.*?([A-Z][A-Z\s]{2,20})[®™]?',
                    first_page_text,
                    re.DOTALL
                )
                if highlight_match:
                    name = highlight_match.group(1).strip().title()
                    if len(name) > 3 and not _is_section_header_line(name):
                        return name

                lines = [l.strip() for l in first_page_text.split('\n')]

                # Pattern 3: a Title-Case product name line — checked before
                # the ALL-CAPS heuristic below so a boxed warning's all-caps
                # heading can't shadow a Title-Case product name that
                # appears earlier on the page (e.g. older generic-drug
                # labels that print the warning before the name).
                for line in lines[:20]:
                    if not line or _is_section_header_line(line):
                        continue
                    if 4 <= len(line) <= 60 and _looks_like_product_name_line(line):
                        clean = re.sub(r'[®™©]', '', line).strip()
                        clean = _strip_dosage_form_suffix(clean)
                        if clean and len(clean) > 2:
                            return clean

                # Pattern 4: first prominent ALL-CAPS line on the page
                for line in lines[:20]:
                    if not line or len(line) <= 3 or len(line) >= 40:
                        continue
                    if _is_section_header_line(line):
                        continue
                    upper_ratio = sum(1 for c in line if c.isupper()) / max(len(line.replace(' ', '')), 1)
                    if upper_ratio > 0.6:
                        clean = re.sub(r'[®™©]', '', line).strip()
                        clean = re.sub(r'\s+', ' ', clean)
                        if clean and len(clean) > 2:
                            return clean.title()

    except Exception:
        pass

    # Fallback: use filename
    return extract_drug_name_from_filename(os.path.basename(filepath))


def extract_drug_name_from_filename(filename: str) -> str:
    """
    Guesses the drug name from the PDF filename.
    e.g. 'ibuprofen_prescribing_info.pdf' -> 'Ibuprofen'
         'rinvoq_pi.pdf' -> 'Rinvoq Pi'
    """
    name = os.path.splitext(filename)[0]       # remove .pdf
    # Remove common suffixes
    for suffix in ['_pi', '_prescribing_info', '_prescribing_information',
                   '_label', '_drug_label', '_fda', '_package_insert']:
        name = name.replace(suffix, '')
    name = name.replace("_", " ").replace("-", " ")  # underscores/hyphens to spaces
    return name.strip().title()                 # Title Case


def load_single_pdf(filepath: str) -> list[dict]:
    """
    Processes a single PDF file and returns a list of chunks.
    Each chunk is a dict with: text, drug_name, page_number, source_file

    Used for incremental indexing when a new PDF is uploaded.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF not found: {filepath}")

    filename = os.path.basename(filepath)
    drug_name = extract_drug_name_from_content(filepath)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []
    try:
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = extract_page_text(page)

                if not page_text or len(page_text.strip()) < 50:
                    continue

                page_chunks = splitter.split_text(page_text)

                for chunk in page_chunks:
                    chunks.append({
                        "text": chunk,
                        "drug_name": drug_name,
                        "page_number": page_num,
                        "source_file": filename,
                        "total_pages": total_pages
                    })

    except Exception as e:
        print(f"Error reading {filename}: {e}")
        raise

    print(f"Processed {filename}: {drug_name} — {len(chunks)} chunks from {total_pages if chunks else 0} pages")
    return chunks


def load_pdfs(pdf_folder: str) -> list[dict]:
    """
    Reads all PDFs in the given folder.
    Returns a list of chunks, each chunk is a dict:
    {
        "text": "...the chunk text...",
        "drug_name": "Ibuprofen",
        "page_number": 4,
        "source_file": "ibuprofen.pdf",
        "total_pages": 20
    }
    """
    all_chunks = []

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in", pdf_folder)
        return []

    for filename in pdf_files:
        filepath = os.path.join(pdf_folder, filename)
        try:
            chunks = load_single_pdf(filepath)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks