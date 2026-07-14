import os
import json
import pdfplumber
import numpy as np
from pdf2image import convert_from_path
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm

CORE_BASELINES_DIR = r"core_baselines"
BROADER_CORPUS_DIR = r"broader_corpus"

BASELINE_TEXT_DIR  = r"extracted_text\core_baselines"
BROADER_TEXT_DIR   = r"extracted_text\broader_corpus"

# If a page returns fewer than this many characters via pdfplumber,
TEXT_LAYER_MIN_CHARS = 50

ocr_engine = RapidOCR()

def extract_page_with_fallback(page, page_num: int, pdf_path: str) -> tuple[str, str]:
    """
    Try pdfplumber text extraction first.
    If the page yields < TEXT_LAYER_MIN_CHARS, fall back to RapidOCR.
    Returns (text, method_used).
    """
    text = page.extract_text() or ""
    if len(text.strip()) >= TEXT_LAYER_MIN_CHARS:
        return text.strip(), "pdfplumber"

    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_num + 1,
            last_page=page_num + 1,
            dpi=300,
        )
        if not images:
            return "", "ocr_failed"

        img_array = np.array(images[0])
        result, _ = ocr_engine(img_array)

        if result:
            ocr_text = "\n".join([line[1] for line in result])
            return ocr_text.strip(), "rapidocr"
        return "", "ocr_empty"

    except Exception as e:
        print(f"      ⚠️  OCR failed on page {page_num + 1}: {e}")
        return "", "ocr_error"

def extract_pdf(pdf_path: str) -> dict:
    pages_text       = []
    method_log       = []
    pdfplumber_count = 0
    ocr_count        = 0
    failed_count     = 0

    filename = os.path.basename(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        with tqdm(
            total=total_pages,
            desc=f"  {filename[:45]}",
            unit="pg",
            ncols=80,
            colour="green",
        ) as pbar:
            for i, page in enumerate(pdf.pages):
                text, method = extract_page_with_fallback(page, i, pdf_path)
                pages_text.append(text)
                method_log.append({"page": i + 1, "method": method})

                if method == "pdfplumber":
                    pdfplumber_count += 1
                elif method == "rapidocr":
                    ocr_count += 1
                else:
                    failed_count += 1

                # Show current method in the progress bar suffix
                pbar.set_postfix({"last": method}, refresh=True)
                pbar.update(1)

    full_text  = "\n\n".join(pages_text)
    word_count = len(full_text.split())

    return {
        "text":             full_text,
        "word_count":       word_count,
        "total_pages":      total_pages,
        "pdfplumber_pages": pdfplumber_count,
        "ocr_pages":        ocr_count,
        "failed_pages":     failed_count,
        "method_log":       method_log,
    }

def process_directory(input_dir: str, output_dir: str, label: str):
    """
    Walk input_dir recursively, extract all PDFs, save .txt files
    and a per-file metadata JSON to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary = []

    for root, _, files in os.walk(input_dir):
        for filename in sorted(files):
            if not filename.endswith(".pdf"):
                continue

            pdf_path    = os.path.join(root, filename)
            relative    = os.path.relpath(root, input_dir)
            out_subdir  = os.path.join(output_dir, relative)
            os.makedirs(out_subdir, exist_ok=True)

            stem     = os.path.splitext(filename)[0]
            txt_path = os.path.join(out_subdir, stem + ".txt")
            log_path = os.path.join(out_subdir, stem + "_extraction_log.json")

            print(f"\n  📄 {relative}/{filename}")

            try:
                result = extract_pdf(pdf_path)
            except Exception as e:
                print(f"     ❌ Skipped — corrupted or invalid PDF: {e}")
                summary.append({
                                "source_pdf": pdf_path,
                                "output_txt": None,
                                "word_count": 0,
                                "total_pages": 0,
                                "pdfplumber_pages": 0,
                                "ocr_pages": 0,
                                "failed_pages": 0,
                                "error": str(e),
                                })
                continue

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

            log = {
                "source_pdf":       pdf_path,
                "output_txt":       txt_path,
                "word_count":       result["word_count"],
                "total_pages":      result["total_pages"],
                "pdfplumber_pages": result["pdfplumber_pages"],
                "ocr_pages":        result["ocr_pages"],
                "failed_pages":     result["failed_pages"],
                "method_log":       result["method_log"],
            }
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2)

            summary.append(log)

            print(f"     ✅ {result['word_count']} words extracted | "
                  f"{result['total_pages']} pages "
                  f"({result['pdfplumber_pages']} text-layer, "
                  f"{result['ocr_pages']} OCR, "
                  f"{result['failed_pages']} failed)")

    summary_path = os.path.join(output_dir, f"_{label}_extraction_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    total_words = sum(r["word_count"] for r in summary)
    total_ocr   = sum(r["ocr_pages"] for r in summary)
    total_pages = sum(r["total_pages"] for r in summary)
    print(f"\n{'═'*55}")
    print(f"  {label} complete")
    print(f"  Files processed : {len(summary)}")
    print(f"  Total pages     : {total_pages}")
    print(f"  OCR pages       : {total_ocr}")
    print(f"  Total words     : {total_words:,}")
    print(f"{'═'*55}")

    return summary

if __name__ == "__main__":

    print("📖 Extracting CORE BASELINES (Stages 2–4)...\n")
    process_directory(
        input_dir  = CORE_BASELINES_DIR,
        output_dir = BASELINE_TEXT_DIR,
        label      = "Core Baselines",
    )

    print("\n📖 Extracting BROADER CORPUS (Stage 6)...\n")
    process_directory(
        input_dir  = BROADER_CORPUS_DIR,
        output_dir = BROADER_TEXT_DIR,
        label      = "Broader Corpus",
    )

    print("\n✅ All extractions complete.")
    print(f"   Baseline texts → {BASELINE_TEXT_DIR}")
    print(f"   Broader texts  → {BROADER_TEXT_DIR}")