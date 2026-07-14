import shutil
import os

HISTORICAL_CORPUS_DIR = r"historical corpus"
CORE_BASELINES_DIR    = r"core_baselines"
BROADER_CORPUS_DIR    = r"broader_corpus"

CORE_BASELINE_FILES = {
    "Air Pollution":     "Air (Prevention and Control of Pollution) Act, 1981.pdf",
    "Data Protection":   "Digital Personal Data Protection Act, 2023.pdf",
    "National Security": "National Security Act, 1980.pdf",
}

core_values = set(CORE_BASELINE_FILES.values())
os.makedirs(CORE_BASELINES_DIR, exist_ok=True)
os.makedirs(BROADER_CORPUS_DIR, exist_ok=True)

for domain_folder in os.listdir(HISTORICAL_CORPUS_DIR):
    domain_path = os.path.join(HISTORICAL_CORPUS_DIR, domain_folder)
    if not os.path.isdir(domain_path):
        continue
    broader_domain_dir = os.path.join(BROADER_CORPUS_DIR, domain_folder)
    os.makedirs(broader_domain_dir, exist_ok=True)
    for filename in os.listdir(domain_path):
        if not filename.endswith(".pdf"):
            continue
        src = os.path.join(domain_path, filename)
        if filename in core_values:
            shutil.copy2(src, os.path.join(CORE_BASELINES_DIR, filename))
            print(f"  ✅ [CORE]    {filename}")
        else:
            shutil.copy2(src, os.path.join(broader_domain_dir, filename))
            print(f"  📁 [BROADER] {domain_folder}/{filename}")

print("\nSegregation done — now running extraction...\n")