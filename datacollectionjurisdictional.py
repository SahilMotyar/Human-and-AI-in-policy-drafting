"""
Stage 5 — Data Collection Script
Downloads and prepares all corpora needed for Jurisdictional Drift analysis.
Run this ONCE before running jurisdictionaldrift.py

What this downloads:
1. Indian Legal Corpus — from HuggingFace viber1/indian-law-dataset
2. US Legal Corpus    — from HuggingFace pile-of-law/pile-of-law (us_code subset)
3. Verifies GDPR file is present (you add this manually)
4. Verifies Sarvam outputs are accessible
"""

import os
import re
import json
import pandas as pd
from pathlib import Path

# ── Output directories ──────────────────────────────────────────────────────
BASE_DIR         = r"extracted_text\jurisdictional"
INDIAN_DIR       = os.path.join(BASE_DIR, "indian_corpus")
US_DIR           = os.path.join(BASE_DIR, "us_corpus")
GDPR_DIR         = os.path.join(BASE_DIR, "gdpr")

for d in [BASE_DIR, INDIAN_DIR, US_DIR, GDPR_DIR]:
    os.makedirs(d, exist_ok=True)

print("Output directories created:")
print(f"  Indian corpus → {INDIAN_DIR}")
print(f"  US corpus     → {US_DIR}")
print(f"  GDPR          → {GDPR_DIR}")

# ── 1. Indian Legal Corpus ───────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Downloading Indian Legal Corpus")
print("="*60)

def download_indian_corpus(output_dir, max_docs=500, min_words=200):
    """
    Downloads from viber1/indian-law-dataset on HuggingFace.
    Filters to substantive documents only (min_words threshold).
    Saves each document as a separate .txt file.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] datasets library not installed.")
        print("        Run: pip install datasets")
        return 0

    print("Loading viber1/indian-law-dataset from HuggingFace...")
    print("(This may take a few minutes on first download)")

    try:
        dataset = load_dataset(
            "viber1/indian-law-dataset",
            split="train",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"[ERROR] Could not load dataset: {e}")
        print("Trying alternate split name...")
        try:
            dataset = load_dataset(
                "viber1/indian-law-dataset",
                trust_remote_code=True
            )
            # Get first available split
            split_name = list(dataset.keys())[0]
            dataset = dataset[split_name]
            print(f"  Loaded split: '{split_name}'")
        except Exception as e2:
            print(f"[ERROR] Failed completely: {e2}")
            return 0

    print(f"  Dataset loaded: {len(dataset)} total records")
    print(f"  Columns: {dataset.column_names}")

    # Identify the text column
    text_col = None
    for candidate in ['text', 'content', 'judgment', 'document', 'body', 'Response', 'response']:
        if candidate in dataset.column_names:
            text_col = candidate
            break

    if text_col is None:
        # Use first string column
        for col in dataset.column_names:
            if dataset.features[col].dtype == 'string':
                text_col = col
                break

    if text_col is None:
        print(f"[ERROR] No text column found. Columns: {dataset.column_names}")
        return 0

    print(f"  Using text column: '{text_col}'")

    saved = 0
    skipped = 0

    for i, record in enumerate(dataset):
        if saved >= max_docs:
            break

        text = str(record.get(text_col, "")).strip()

        # Filter short documents
        if len(text.split()) < min_words:
            skipped += 1
            continue

        # Clean text
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            skipped += 1
            continue

        filename = os.path.join(output_dir, f"indian_legal_{saved:04d}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)

        saved += 1

        if saved % 50 == 0:
            print(f"  Saved {saved}/{max_docs} documents...")

    print(f"\n  Indian corpus complete:")
    print(f"    Saved:   {saved} documents")
    print(f"    Skipped: {skipped} (too short)")
    return saved

indian_count = download_indian_corpus(INDIAN_DIR, max_docs=500)

# ── 2. US Legal Corpus ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Downloading US Legal Corpus")
print("="*60)

def download_us_corpus(output_dir, max_docs=500, min_words=200):
    """
    Downloads US Code subset from pile-of-law/pile-of-law on HuggingFace.
    Uses streaming to avoid downloading the full multi-TB dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] datasets library not installed.")
        print("        Run: pip install datasets")
        return 0

    print("Loading pile-of-law/pile-of-law (us_code subset) via streaming...")
    print("(Streaming mode — no full download required)")

    # Priority subsets — most relevant to legal drafting
    subsets_to_try = [
        ("pile-of-law/pile-of-law", "us_code"),
        ("pile-of-law/pile-of-law", "cfr"),           # Code of Federal Regulations
        ("pile-of-law/pile-of-law", "federal_register"),
    ]

    saved = 0

    for dataset_name, subset in subsets_to_try:
        if saved >= max_docs:
            break

        remaining = max_docs - saved
        print(f"\n  Trying subset: '{subset}' (need {remaining} more docs)")

        try:
            dataset = load_dataset(
                dataset_name,
                subset,
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            # Identify text column
            text_col = None
            for candidate in ['text', 'content', 'document', 'body']:
                # Can't check columns without iterating in streaming mode
                # Try 'text' first as it's most common in pile-of-law
                text_col = 'text'
                break

            subset_saved = 0
            subset_skipped = 0

            for record in dataset:
                if subset_saved >= remaining:
                    break

                # Try multiple column names
                text = ""
                for col in ['text', 'content', 'document', 'body']:
                    if col in record and record[col]:
                        text = str(record[col]).strip()
                        break

                if not text or len(text.split()) < min_words:
                    subset_skipped += 1
                    continue

                # Clean text
                text = text.encode('ascii', 'ignore').decode('ascii')
                text = re.sub(r'\s+', ' ', text).strip()

                if not text:
                    subset_skipped += 1
                    continue

                filename = os.path.join(
                    output_dir,
                    f"us_legal_{subset}_{saved:04d}.txt"
                )
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)

                saved += 1
                subset_saved += 1

                if saved % 50 == 0:
                    print(f"    Saved {saved}/{max_docs} total...")

            print(f"  Subset '{subset}': saved {subset_saved}, skipped {subset_skipped}")

        except Exception as e:
            print(f"  [warn] Subset '{subset}' failed: {e}")
            print(f"         Trying next subset...")
            continue

    print(f"\n  US corpus complete: {saved} documents saved")
    return saved

us_count = download_us_corpus(US_DIR, max_docs=500)

# ── 3. GDPR Verification ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: GDPR File Verification")
print("="*60)

gdpr_files = list(Path(GDPR_DIR).glob("*.txt"))

if gdpr_files:
    for f in gdpr_files:
        word_count = len(f.read_text(encoding='utf-8', errors='ignore').split())
        print(f"  [OK] Found: {f.name} ({word_count:,} words)")
else:
    print("  [MISSING] No GDPR .txt file found in:")
    print(f"    {GDPR_DIR}")
    print()
    print("  ACTION REQUIRED:")
    print("  1. Download GDPR PDF from:")
    print("     https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679")
    print("  2. Convert to text using your existing OCR pipeline")
    print(f"  3. Save as: {GDPR_DIR}\\gdpr_full_text.txt")
    print()
    print("  OR run this quick download (plain text version):")
    print("""
    import urllib.request
    url = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679"
    # Use requests + BeautifulSoup to scrape the text content
    # Or use the PDF approach with pdfplumber
    """)

# ── 4. Sarvam Output Verification ────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Sarvam Output Verification")
print("="*60)

MASTER_DF_PATH = r"result\archive\master_dataframe.csv"

try:
    df = pd.read_csv(MASTER_DF_PATH)
    print(f"Loaded master_dataframe.csv: {len(df)} total rows")
    print(f"\nAll models in dataset:")
    print(df['model'].value_counts().to_string())

    sarvam_df = df[df['model'].str.contains('sarvam', case=False, na=False)]
    print(f"\nSarvam rows: {len(sarvam_df)}")

    if len(sarvam_df) > 0:
        print("\nSarvam breakdown:")
        print(sarvam_df.groupby(['domain', 'condition']).size().to_string())

        # Verify files are accessible
        accessible = 0
        missing = 0
        for _, row in sarvam_df.iterrows():
            if pd.notna(row['file']) and os.path.exists(row['file']):
                accessible += 1
            else:
                missing += 1

        print(f"\nFile accessibility:")
        print(f"  Accessible: {accessible}")
        print(f"  Missing:    {missing}")

        if missing > 0:
            print("\n  [warn] Some Sarvam files are missing.")
            print("         Check file paths in master_dataframe.csv")
    else:
        print("[warn] No Sarvam rows found in master_dataframe.csv")
        print("       Check model column name — tried 'sarvam' (case insensitive)")

except FileNotFoundError:
    print(f"[ERROR] master_dataframe.csv not found at:")
    print(f"  {MASTER_DF_PATH}")

# ── Supplement Indian corpus with your existing broader corpus ────────────
print("\nSupplementing Indian corpus with existing broader corpus files...")

EXISTING_BROADER = [
    r"extracted_text\broader_corpus\DPDP",
    r"extracted_text\broader_corpus\Air pollution",
    r"extracted_text\broader_corpus\National security",
]

supplement_count = 0
for folder in EXISTING_BROADER:
    if not os.path.exists(folder):
        print(f"  [warn] Folder not found: {folder}")
        continue
    for fname in os.listdir(folder):
        if not fname.endswith('.txt'):
            continue
        src = os.path.join(folder, fname)
        dst = os.path.join(
            INDIAN_DIR,
            f"existing_broader_{supplement_count:04d}_{fname}"
        )
        import shutil
        shutil.copy2(src, dst)
        supplement_count += 1

print(f"  Copied {supplement_count} existing broader corpus files into Indian corpus")

# ── 5. Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("COLLECTION SUMMARY")
print("="*60)

indian_files = list(Path(INDIAN_DIR).glob("*.txt"))
us_files     = list(Path(US_DIR).glob("*.txt"))
gdpr_files   = list(Path(GDPR_DIR).glob("*.txt"))

print(f"\n  Indian corpus: {len(indian_files)} files")
print(f"  US corpus:     {len(us_files)} files")
print(f"  GDPR:          {len(gdpr_files)} files {'[OK]' if gdpr_files else '[MISSING — add manually]'}")

indian_words = sum(
    len(f.read_text(encoding='utf-8', errors='ignore').split())
    for f in indian_files
)
us_words = sum(
    len(f.read_text(encoding='utf-8', errors='ignore').split())
    for f in us_files
)

print(f"\n  Indian corpus words: {indian_words:,}")
print(f"  US corpus words:     {us_words:,}")
print(f"\n  Minimum recommended: 100,000 words per corpus for stable TF-IDF")

if indian_words < 100000:
    print(f"  [warn] Indian corpus may be too small — consider increasing max_docs")
if us_words < 100000:
    print(f"  [warn] US corpus may be too small — consider increasing max_docs")

# Save corpus paths for use in Stage 5 script
config = {
    "indian_corpus_dir": INDIAN_DIR,
    "us_corpus_dir":     US_DIR,
    "gdpr_dir":          GDPR_DIR,
    "indian_doc_count":  len(indian_files),
    "us_doc_count":      len(us_files),
    "gdpr_present":      len(gdpr_files) > 0,
    "indian_word_count": indian_words,
    "us_word_count":     us_words,
}

config_path = os.path.join(BASE_DIR, "corpus_config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n  Config saved to: {config_path}")
print("  (Stage 5 script will read this automatically)")

print("\n" + "="*60)
if gdpr_files:
    print("ALL READY — run jurisdictionaldrift.py next")
else:
    print("ALMOST READY — add GDPR text file then run jurisdictionaldrift.py")
print("="*60)