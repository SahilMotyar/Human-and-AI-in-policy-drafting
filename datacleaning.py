import os
import re
import json
import pandas as pd

BASE_DIR = r"."

INPUT_DIRS = {
    "chatgpt": os.path.join(BASE_DIR, "output", "chatgpt_policy_drafts"),
    "gemini":  os.path.join(BASE_DIR, "output", "gemini_policy_drafts"),
    "sarvam":  os.path.join(BASE_DIR, "output", "sarvam_policy_drafts"),
}

BASELINE_DIR  = os.path.join(BASE_DIR, "extracted_text", "core_baselines")
BROADER_DIR   = os.path.join(BASE_DIR, "extracted_text", "broader_corpus")
EXCLUSION_LOG = os.path.join(BASE_DIR, "result", "exclusion_register.json")
MASTER_CSV    = os.path.join(BASE_DIR, "result", "master_dataframe.csv")

SIZE_THRESHOLD_KB = 3

DOMAINS = {
    "air":      "Air Pollution",
    "dpdp":     "Data Protection",
    "national": "National Security",
}

CONDITIONS = {
    "innovation":     "Innovation",
    "innovatino":     "Innovation",
    "unconstrained":  "Unconstrained",
    "unconstratined": "Unconstrained",
    "status quo":     "Status Quo",
    "statusquo":      "Status Quo",
    "status_quo":     "Status Quo",
}

MODELS = {
    "chatgpt": "ChatGPT",
    "gpt":     "ChatGPT",
    "gemini":  "Gemini",
    "gemino":  "Gemini",
    "gemini_": "Gemini",
    "sarvam":  "Sarvam",
}

BASELINE_DOMAIN_MAP = {
    "air (prevention and control of pollution) act, 1981": "Air Pollution",
    "digital personal data protection act, 2023":          "Data Protection",
    "national security act, 1980":                         "National Security",
}

BROADER_FOLDER_MAP = {
    "air pollution": "Air Pollution",
    "dpdp":          "Data Protection",
    "national security": "National Security",
}

PRIMARY_MODELS = {"ChatGPT", "Gemini"}

REFUSAL_PHRASES = [
    "i'm sorry", "i cannot", "i am unable", "as an ai",
    "i don't think i can", "this request", "i must decline",
    "i apologize", "cannot assist",
]

PREAMBLE_PHRASES = [
    "sure!", "of course!", "certainly!", "here is", "here's",
    "below is", "as requested", "happy to help",
]

def extract_text(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, 'r', encoding='latin-1') as f:
            return f.read()

def parse_filename(filename: str, folder_model_key: str = None) -> dict | None:
    stem = os.path.splitext(filename)[0].lower()

    domain = None
    for key, label in DOMAINS.items():
        if stem.startswith(key):
            domain = label
            break
    if not domain:
        return None

    condition = None
    for key, label in CONDITIONS.items():
        if key in stem:
            condition = label
            break
    if not condition:
        return None

    iter_match = re.search(r'iteration\s*(\d+)', stem)
    iteration  = int(iter_match.group(1)) if iter_match else None

    model = None
    for key, label in MODELS.items():
        if key in stem:
            model = label
            break
    if not model and folder_model_key:
        model = MODELS.get(folder_model_key)
    if not model:
        return None

    return {"domain": domain, "condition": condition,
            "iteration": iteration, "model": model}

def flag_issues(text: str, file_size_kb: float) -> list[str]:
    issues = []
    lower  = text.lower()
    if file_size_kb < SIZE_THRESHOLD_KB:
        issues.append(f"under_{SIZE_THRESHOLD_KB}KB")
    for phrase in REFUSAL_PHRASES:
        if phrase in lower:
            issues.append("refusal_detected")
            break
    for phrase in PREAMBLE_PHRASES:
        if lower.strip().startswith(phrase):
            issues.append("conversational_preamble")
            break
    if len(text.split()) < 100:
        issues.append("very_short_output")
    return issues

def build_ai_corpus(records, exclusions):
    print("📝 Loading AI drafts...\n")
    for model_key, folder in INPUT_DIRS.items():
        if not os.path.isdir(folder):
            print(f"  ⚠️  Folder not found, skipping: {folder}")
            continue

        for root, _, files in os.walk(folder):
            for filename in sorted(files):
                if not filename.endswith(".txt"):
                    continue

                txt_path     = os.path.join(root, filename)
                file_size_kb = os.path.getsize(txt_path) / 1024

                parsed = parse_filename(filename, folder_model_key=model_key)
                if not parsed:
                    print(f"  ⚠️  Could not parse filename: {filename}")
                    exclusions.append({"file": txt_path, "reason": "unparseable_filename"})
                    continue

                try:
                    text = extract_text(txt_path)
                except Exception as e:
                    print(f"  ❌ Read error: {filename} — {e}")
                    exclusions.append({"file": txt_path, "reason": f"read_error: {e}"})
                    continue

                word_count  = len(text.split())
                issues      = flag_issues(text, file_size_kb)
                corpus_type = "Primary" if parsed["model"] in PRIMARY_MODELS else "Supplementary"
                is_partial  = (
                    parsed["model"]  == "Sarvam"
                    and parsed["domain"] == "National Security"
                    and "refusal_detected" in issues
                )

                records.append({
                    "file":         txt_path,
                    "filename":     filename,
                    "model":        parsed["model"],
                    "domain":       parsed["domain"],
                    "condition":    parsed["condition"],
                    "iteration":    parsed["iteration"],
                    "corpus_type":  corpus_type,
                    "is_partial":   is_partial,
                    "word_count":   word_count,
                    "file_size_kb": round(file_size_kb, 2),
                    "issues":       "|".join(issues) if issues else "none",
                    "text":         text,
                    "source":       "ai_draft",
                })

                if issues:
                    exclusions.append({
                        "file": txt_path, "issues": issues,
                        "word_count": word_count, "file_size_kb": round(file_size_kb, 2),
                    })

                status = "⚠️ " if issues else "✅"
                print(f"  {status} [{corpus_type}] {parsed['model']:8s} | "
                      f"{parsed['domain']:20s} | {parsed['condition']:15s} | "
                      f"iter {parsed['iteration']} | {word_count} words")

def build_baseline_corpus(records, exclusions):
    print("\n📜 Loading core baselines...\n")
    if not os.path.isdir(BASELINE_DIR):
        print(f"  ⚠️  Baseline dir not found: {BASELINE_DIR}")
        return

    for filename in sorted(os.listdir(BASELINE_DIR)):
        if not filename.endswith(".txt"):
            continue

        txt_path     = os.path.join(BASELINE_DIR, filename)
        file_size_kb = os.path.getsize(txt_path) / 1024
        stem         = os.path.splitext(filename)[0].lower()

        domain = BASELINE_DOMAIN_MAP.get(stem)
        if not domain:
            print(f"  ⚠️  Unrecognised baseline file: {filename}")
            exclusions.append({"file": txt_path, "reason": "unrecognised_baseline"})
            continue

        try:
            text = extract_text(txt_path)
        except Exception as e:
            print(f"  ❌ Read error: {filename} — {e}")
            exclusions.append({"file": txt_path, "reason": f"read_error: {e}"})
            continue

        word_count = len(text.split())

        records.append({
            "file":         txt_path,
            "filename":     filename,
            "model":        "Baseline",
            "domain":       domain,
            "condition":    None,       # baselines have no prompt condition
            "iteration":    None,
            "corpus_type":  "Baseline",
            "is_partial":   False,
            "word_count":   word_count,
            "file_size_kb": round(file_size_kb, 2),
            "issues":       "none",
            "text":         text,
            "source":       "core_baseline",
        })

        print(f"  ✅ [Baseline] {domain:20s} | {filename} | {word_count} words")

def build_broader_corpus(records, exclusions):
    print("\n📚 Loading broader corpus...\n")
    if not os.path.isdir(BROADER_DIR):
        print(f"  ⚠️  Broader corpus dir not found: {BROADER_DIR}")
        return

    for subfolder in sorted(os.listdir(BROADER_DIR)):
        subfolder_path = os.path.join(BROADER_DIR, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        domain = BROADER_FOLDER_MAP.get(subfolder.lower())
        if not domain:
            print(f"  ⚠️  Unrecognised subfolder: {subfolder}")
            continue

        for filename in sorted(os.listdir(subfolder_path)):
            if not filename.endswith(".txt"):
                continue
            if filename.endswith("_extraction_log.json") or filename.startswith("_"):
                continue

            txt_path     = os.path.join(subfolder_path, filename)
            file_size_kb = os.path.getsize(txt_path) / 1024

            try:
                text = extract_text(txt_path)
            except Exception as e:
                print(f"  ❌ Read error: {filename} — {e}")
                exclusions.append({"file": txt_path, "reason": f"read_error: {e}"})
                continue

            word_count = len(text.split())

            records.append({
                "file":         txt_path,
                "filename":     filename,
                "model":        "Broader",
                "domain":       domain,
                "condition":    None,
                "iteration":    None,
                "corpus_type":  "Broader",
                "is_partial":   False,
                "word_count":   word_count,
                "file_size_kb": round(file_size_kb, 2),
                "issues":       "none",
                "text":         text,
                "source":       "broader_corpus",
            })

            print(f"  ✅ [Broader] {domain:20s} | {filename} | {word_count} words")

def verify_balance(df: pd.DataFrame):
    print("\n" + "═" * 60)
    print("PRIMARY CORPUS BALANCE CHECK")
    print("═" * 60)
    primary = df[df["corpus_type"] == "Primary"]
    print(primary.groupby(["model", "domain", "condition"])
          .size().reset_index(name="count").to_string(index=False))
    total    = len(primary)
    expected = 2 * 3 * 3 * 10
    print(f"\nTotal primary drafts: {total}  →  "
          f"{'✅ Balanced' if total == expected else f'⚠️  Expected {expected}, got {total}'}")

    print("\n" + "═" * 60)
    print("SUPPLEMENTARY CORPUS (SARVAM)")
    print("═" * 60)
    sarvam = df[df["corpus_type"] == "Supplementary"]
    print(sarvam.groupby(["domain", "condition", "is_partial"])
          .size().reset_index(name="count").to_string(index=False))

    print("\n" + "═" * 60)
    print("BASELINE CORPUS")
    print("═" * 60)
    baseline = df[df["corpus_type"] == "Baseline"]
    print(baseline[["domain", "filename", "word_count"]].to_string(index=False))

    print("\n" + "═" * 60)
    print("BROADER CORPUS")
    print("═" * 60)
    broader = df[df["corpus_type"] == "Broader"]
    print(broader.groupby("domain").agg(
        files=("filename", "count"),
        total_words=("word_count", "sum")
    ).to_string())

    print("\n" + "═" * 60)
    print("WORD COUNT SUMMARY (AI drafts)")
    print("═" * 60)
    ai = df[df["corpus_type"].isin(["Primary", "Supplementary"])]
    print(ai.groupby(["model", "domain", "condition"])["word_count"]
          .describe()[["mean", "min", "max"]].to_string())

def save_outputs(df: pd.DataFrame, exclusions: list[dict]):
    df.drop(columns=["text"]).to_csv(MASTER_CSV, index=False)
    print(f"\n💾 Master dataframe saved → {MASTER_CSV}")
    with open(EXCLUSION_LOG, "w", encoding="utf-8") as f:
        json.dump(exclusions, f, indent=2)
    print(f"📋 Exclusion register saved → {EXCLUSION_LOG}")

if __name__ == "__main__":
    records    = []
    exclusions = []

    build_ai_corpus(records, exclusions)
    build_baseline_corpus(records, exclusions)
    build_broader_corpus(records, exclusions)

    master_df = pd.DataFrame(records)

    if master_df.empty:
        print("\n❌ No files parsed. Check your paths.")
    else:
        verify_balance(master_df)
        save_outputs(master_df, exclusions)
        print(f"\n✅ Stage 1 complete. Total records: {len(master_df)}")
        by_type = master_df.groupby("corpus_type").size()
        print(by_type.to_string())