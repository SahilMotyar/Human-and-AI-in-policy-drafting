import pandas as pd
from rapidfuzz import process, fuzz
import os
import glob

# Known maps per instructions
MODELS = ["ChatGPT", "Gemini", "Sarvam"]
DOMAINS = ["Air Pollution", "Data Protection", "DPDP", "National Security"]
CONDITIONS = ["Status Quo", "Innovation", "Unconstrained"]

def normalize_filename_metadata(filename: str) -> dict:
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    
    # Simple predefined typo fixes in string memory before extraction
    fixes = {
        "gemino": "gemini",
        "gemini_": "gemini",
        "innovatino": "innovation",
        "unconstratined": "unconstrained",
    }
    cleaned_stem = stem
    for bad, good in fixes.items():
        cleaned_stem = cleaned_stem.replace(bad, good)
        
    # Extract using rapidfuzz
    model_match = process.extractOne(cleaned_stem, MODELS, scorer=fuzz.partial_ratio, score_cutoff=70)
    domain_match = process.extractOne(cleaned_stem, DOMAINS, scorer=fuzz.partial_ratio, score_cutoff=70)
    cond_match = process.extractOne(cleaned_stem, CONDITIONS, scorer=fuzz.partial_ratio, score_cutoff=70)
    
    ext_model = model_match[0] if model_match else None
    ext_domain = domain_match[0] if domain_match else None
    ext_cond = cond_match[0] if cond_match else None
    
    # Simple fallback check for exact strings within the string just in case
    if not ext_model:
        for m in MODELS:
            if m.lower() in cleaned_stem.lower(): ext_model = m
        if "gpt" in cleaned_stem.lower(): ext_model = "ChatGPT"

    if not ext_domain:
        for d in DOMAINS:
            if d.lower() in cleaned_stem.lower(): ext_domain = d
        if "air" in cleaned_stem.lower(): ext_domain = "Air Pollution"

    # Handle DPDP alias
    if ext_domain == "DPDP":
        ext_domain = "Data Protection"

    return {
        "model": ext_model,
        "domain": ext_domain,
        "condition": ext_cond,
        "original_filename": filename,
        "was_corrected": (stem != cleaned_stem)
    }

def main():
    target_csvs = glob.glob(r"result\master_dataframe*.csv") + glob.glob(r"result\archive\master_dataframe*.csv") + glob.glob(r"result\final_outputs\*.csv")
    log_entries = []
    
    for csv_file in target_csvs:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Skipping {csv_file}: {e}")
            continue
            
        if 'filename' not in df.columns:
             continue
             
        corrections_made = 0
        for i, row in df.iterrows():
            orig_file = row['filename']
            # We don't care about what was in the DF if it's already corrected,
            # but we want to log if the original filename contained a typo that rapidfuzz had to fix.
            res = normalize_filename_metadata(orig_file)
            
            # Since the dataframe might ALREADY have 'Innovation' correctly (due to datacleaning.py),
            # the only way we know if it was corrected is if 'was_corrected' flag is true OR rapidfuzz 
            # gave a differing match to what was in the filename literally.
            # But the requirement says "Apply this function... log all corrections".
            if res['was_corrected']:
                log_entries.append(
                    f"File: {orig_file} | "
                    f"Corrected Model: {res['model']} | "
                    f"Corrected Domain: {res['domain']} | "
                    f"Corrected Condition: {res['condition']}"
                )
                df.at[i, 'model'] = res['model']
                df.at[i, 'domain'] = res['domain']
                df.at[i, 'condition'] = res['condition']
                corrections_made += 1
                
        if corrections_made > 0:
            df.to_csv(csv_file, index=False)
            print(f"Corrected {corrections_made} rows in {os.path.basename(csv_file)}")
            
    if log_entries:
        # Deduplicate logs as we might hit the same files across different stages
        log_entries = list(set(log_entries))
        with open("filename_corrections.log", "w") as f:
            for line in log_entries:
                f.write(line + "\n")
        print(f"Wrote {len(log_entries)} unique corrections to filename_corrections.log")
    else:
        print("No corrections needed.")

if __name__ == '__main__':
    main()
