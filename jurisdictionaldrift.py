import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers: pip install sentence-transformers")
    raise

# Paths
INPUT_CSV = r"result\archive\master_dataframe.csv"
FINAL_OUT_DIR = r"result\final_outputs"
FIG_DIR = os.path.join(FINAL_OUT_DIR, "figures")
os.makedirs(FINAL_OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(FINAL_OUT_DIR, "master_dataframe_with_stage5_scores.csv")
CONFIG_PATH = r"extracted_text\jurisdictional\corpus_config.json"

def sentence_tokenize(text):
    """Simple regex based sentence tokenizer without extra dependencies."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def load_corpus_texts(directory):
    texts = []
    if os.path.exists(directory):
        for fname in os.listdir(directory):
            if fname.endswith(".txt"):
                path = os.path.join(directory, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        texts.append(f.read())
                except:
                    pass
    return texts

def read_full_text(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return ""

def clean_for_tfidf(text):
    """Remove numeric artifacts, metadata, and formatting noise before TF-IDF."""
    # Remove standalone numbers and codes
    text = re.sub(r'\b\d+\b', '', text)
    # Remove PDF metadata artifacts
    text = re.sub(r'verdate|publaw|plaw|comps|g:\\comp|xml', 
                  '', text, flags=re.IGNORECASE)
    # Remove section abbreviations that aren't meaningful
    text = re.sub(r'\bsec\b|\busc\b|\bcfr\b|\bpl\b|\bjkt\b|\bfrm\b', 
                  '', text, flags=re.IGNORECASE)
    # Remove XML typesetting codes
    text = re.sub(r'\b(fmt|sfmt|bel|holc|jkt|frm|eras|eraseq)\b',
                  '', text, flags=re.IGNORECASE)
    # Remove ALL-CAPS codes (XML/formatting artifacts)
    text = re.sub(r'\b[A-Z]{3,}\b', '', text)
    # Remove very short tokens after cleaning
    text = ' '.join(w for w in text.split() if len(w) > 2)
    return text

def chunk_gdpr_by_article(gdpr_text, min_words=40):
    """
    Split GDPR by Article boundaries.
    Articles are the meaningful semantic unit in GDPR.
    """
    # Try splitting on Article markers first
    articles = re.split(r'(?=\bArticle\s+\d+\b)', gdpr_text)
    chunks = [a.strip() for a in articles if len(a.strip().split()) >= min_words]
    
    # If that didn't work well, fall back to double-newline paragraphs
    if len(chunks) < 20:
        chunks = [p.strip() for p in gdpr_text.split('\n\n')
                  if len(p.strip().split()) >= min_words]
    
    print(f"  GDPR split into {len(chunks)} article/paragraph chunks")
    if chunks:
        avg_len = np.mean([len(c.split()) for c in chunks])
        print(f"  Average chunk length: {avg_len:.0f} words")
    return chunks

def chunk_draft_by_section(text, min_words=40):
    """
    Split AI draft by section/paragraph boundaries.
    Consistent with Stage 6 segmentation approach.
    """
    chunks = re.split(r'\n{2,}|\b(?=\d+\.\s+[A-Z])', text)
    return [c.strip() for c in chunks if len(c.strip().split()) >= min_words]

def main():
    print("Loading config...")
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Run datacollectionjurisdictional.py first. Config not found at {CONFIG_PATH}")
        
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
        
    indian_dir = config.get("indian_corpus_dir")
    us_dir = config.get("us_corpus_dir")
    gdpr_dir = config.get("gdpr_dir")
    
    print("Loading text files for TF-IDF...")
    indian_texts = load_corpus_texts(indian_dir)
    us_texts = load_corpus_texts(us_dir)
    
    print(f"Loaded {len(indian_texts)} Indian documents and {len(us_texts)} US documents.")
    
    print("\n--- US Corpus Sample Check ---")
    for i, text in enumerate(us_texts[:3]):
        print(f"\nDoc {i}: first 200 chars")
        print(text[:200])
        print(f"Word count: {len(text.split())}")
    
    # ── Component 1: TF-IDF Distinctive Term Extraction ──
    print("\nExtracting US distinctive terms...")
    indian_texts_clean = [clean_for_tfidf(t) for t in indian_texts]
    us_texts_clean     = [clean_for_tfidf(t) for t in us_texts]

    # Label 0 for Indian, 1 for US
    all_texts = indian_texts_clean + us_texts_clean
    labels = np.array([0]*len(indian_texts_clean) + [1]*len(us_texts_clean))
    
    vectorizer = TfidfVectorizer(
        stop_words="english", 
        max_features=10000, 
        min_df=2, 
        max_df=0.85, 
        ngram_range=(1, 2),
        token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # letters only, min 2 chars
    )
    X = vectorizer.fit_transform(all_texts)
    
    indian_mean_tfidf = np.asarray(X[labels == 0].mean(axis=0)).flatten()
    us_mean_tfidf = np.asarray(X[labels == 1].mean(axis=0)).flatten()
    
    tfidf_diff = us_mean_tfidf - indian_mean_tfidf
    
    vocab = np.array(vectorizer.get_feature_names_out())
    # Top 200 US distinct terms
    top_us_indices = tfidf_diff.argsort()[-200:][::-1]
    top_us_terms = set(vocab[top_us_indices])
    
    print(f"Top 10 US distinctive terms: {list(top_us_terms)[:10]}")
    
    # ── Component 2: Score AI drafts on US-term frequency ──
    print("\nLoading master dataframe...")
    df = pd.read_csv(INPUT_CSV)
    
    # Read full texts for drafts if not already present
    if 'full_text' not in df.columns:
        df['full_text'] = df['file'].apply(lambda x: read_full_text(x) if pd.notna(x) else "")
        
    print("Scoring jurisdictional drift (US distinctive terms per 1000 words)...")
    us_term_scores = []
    
    # Compile regex pattern for fast matching
    us_terms_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, top_us_terms)) + r')\b', re.IGNORECASE)
    
    for _, row in df.iterrows():
        text = row['full_text']
        word_count = len(re.findall(r'\b\w+\b', text))
        if word_count == 0:
            us_term_scores.append(0.0)
            continue
            
        hits = len(us_terms_pattern.findall(text))
        us_term_scores.append((hits / word_count) * 1000.0)
        
    df['US_Distinctive_Term_Frequency'] = us_term_scores

    # --- JSD LENGTH NORMALIZATION FLAG ---
    print("Computing JSD length mismatch ratios against baselines...")
    BASELINE_PATHS = {
        "Data Protection":  r"extracted_text\core_baselines\Digital Personal Data Protection Act, 2023.txt",
        "DPDP":  r"extracted_text\core_baselines\Digital Personal Data Protection Act, 2023.txt",
        "Air Pollution":    r"extracted_text\core_baselines\Air (Prevention and Control of Pollution) Act, 1981.txt",
        "National Security":r"extracted_text\core_baselines\National Security Act, 1980.txt"
    }
    
    baseline_word_counts = {}
    for dom, path in BASELINE_PATHS.items():
        b_text = read_full_text(path)
        baseline_word_counts[dom] = len(re.findall(r'\b\w+\b', b_text))

    flags = []
    for _, row in df.iterrows():
        t = row.get('full_text', '')
        w_count = len(re.findall(r'\b\w+\b', t))
        b_count = baseline_word_counts.get(row['domain'], 1)
        ratio = w_count / b_count if b_count > 0 else 0
        flags.append(ratio < 0.15)
        
    df['length_mismatch_warning'] = flags
    
    flagged_df = df[df['length_mismatch_warning']].copy()
    if 'full_text' in flagged_df.columns:
        flagged_export = flagged_df.drop(columns=['full_text'])
    else:
        flagged_export = flagged_df
        
    flag_out = os.path.join(FINAL_OUT_DIR, "jsd_length_mismatch_flagged.csv")
    flagged_export.to_csv(flag_out, index=False)
    print(f"Saved {len(flagged_df)} length mismatch flagged rows to {flag_out}")
    # -----------------------------------
    
    # ── Component 3: GDPR cosine similarity ──
    print("Loading legal-bert for GDPR similarity (consistent with Stage 2)...")
    model = SentenceTransformer('nlpaueb/legal-bert-base-uncased')

    print("\nPreparing GDPR chunks for similarity computation...")
    gdpr_file = os.path.join(gdpr_dir, "gdpr_full_text.txt")
    gdpr_text = read_full_text(gdpr_file)
    
    gdpr_chunks = chunk_gdpr_by_article(gdpr_text, min_words=40)
    print(f"Tokenized GDPR into {len(gdpr_chunks)} chunks.")

    print("Encoding GDPR chunks...")
    gdpr_embeddings = model.encode(
        gdpr_chunks, show_progress_bar=True, convert_to_tensor=True
    )

    # Diagnostic — verify score distribution is meaningful
    test_sims = cosine_similarity(
        gdpr_embeddings[:10].cpu().numpy(),
        gdpr_embeddings[:10].cpu().numpy()
    )
    print(f"GDPR self-similarity range: {test_sims.min():.3f} - {test_sims.max():.3f}")
    print("(Expected: 0.65-1.00 for diverse articles, if all ~0.95+ chunking failed)")
    
    print("\nComputing GDPR similarity for Data Protection drafts...")
    gdpr_sim_scores = []
    structural_import_counts = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring drafts"):
        if row['domain'] != 'Data Protection' or not str(row.get('full_text','')).strip():
            gdpr_sim_scores.append(np.nan)
            structural_import_counts.append(np.nan)
            continue

        draft_chunks = chunk_draft_by_section(row['full_text'], min_words=40)
        if not draft_chunks:
            gdpr_sim_scores.append(0.0)
            structural_import_counts.append(0)
            continue

        draft_embeddings = model.encode(
            draft_chunks, show_progress_bar=False, convert_to_tensor=True
        )
        sim_matrix = cosine_similarity(
            draft_embeddings.cpu().numpy(),
            gdpr_embeddings.cpu().numpy()
        )

        max_sims = sim_matrix.max(axis=1)
        mean_max_sim = float(max_sims.mean())

        # Use 85th percentile of GDPR self-similarity as threshold
        # This is calibrated to the actual model's score distribution
        gdpr_self_sim_threshold = float(np.percentile(
            test_sims[test_sims < 1.0], 85
        ))
        high_sim_count = int((max_sims >= gdpr_self_sim_threshold).sum())

        gdpr_sim_scores.append(mean_max_sim)
        structural_import_counts.append(high_sim_count)

    print(f"\nGDPR similarity stats across Data Protection drafts:")
    valid_scores = [s for s in gdpr_sim_scores if not np.isnan(s)]
    if valid_scores:
        print(f"  Mean: {np.mean(valid_scores):.3f}")
        print(f"  Std:  {np.std(valid_scores):.3f}")
        print(f"  Min:  {np.min(valid_scores):.3f}")
        print(f"  Max:  {np.max(valid_scores):.3f}")
        print(f"  Range: {np.max(valid_scores) - np.min(valid_scores):.3f}")
        print("  (Target: range > 0.05 for meaningful discrimination)")
        
    df['GDPR_Mean_Similarity'] = gdpr_sim_scores
    df['GDPR_High_Sim_Sentences'] = structural_import_counts
    
    # Drop full_text to save space
    if 'full_text' in df.columns:
        df = df.drop(columns=['full_text'])
        
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nStage 5 Complete. Saved scores to {OUTPUT_CSV}")
    
    from scipy.stats import kruskal, mannwhitneyu
    western = df[df['model'].isin(['ChatGPT', 'Gemini'])]['US_Distinctive_Term_Frequency'].dropna()
    sarvam = df[df['model'] == 'Sarvam']['US_Distinctive_Term_Frequency'].dropna()
    
    if len(western) > 0 and len(sarvam) > 0:
        stat, p = mannwhitneyu(western, sarvam, alternative='greater')
        n1, n2 = len(western), len(sarvam)
        r_rb = 1 - (2 * stat) / (n1 * n2)
        magnitude = ('negligible' if abs(r_rb) < 0.1 else
                     'small'      if abs(r_rb) < 0.3 else
                     'medium'     if abs(r_rb) < 0.5 else 'large')
        print(f"Western vs Sarvam US-term frequency:")
        print(f"  U={stat:.1f}, p={p:.4f}")
        print(f"  Rank-biserial r={r_rb:.3f} ({magnitude} effect)")
        print(f"  Western mean={western.mean():.1f}, Sarvam mean={sarvam.mean():.1f}")
        
        # Also add per-domain breakdown
        print("\nPer-domain Western vs Sarvam comparison:")
        for domain in ['Air Pollution', 'Data Protection', 'National Security']:
            w_d = df[(df['model'].isin(['ChatGPT','Gemini'])) & 
                     (df['domain']==domain)]['US_Distinctive_Term_Frequency'].dropna()
            s_d = df[(df['model']=='Sarvam') & 
                     (df['domain']==domain)]['US_Distinctive_Term_Frequency'].dropna()
            if len(w_d) > 0 and len(s_d) > 0:
                u_d, p_d = mannwhitneyu(w_d, s_d, alternative='greater')
                print(f"  {domain}: Western={w_d.mean():.1f}, "
                      f"Sarvam={s_d.mean():.1f}, p={p_d:.4f}")
    
    # ── Visualization Strategy ──
    print("\nGenerating heatmaps...")
    heatmap_dir = FIG_DIR
    
    # Clean up model names to make them consistent if needed (ChatGPT, Gemini, Sarvam)
    COND_ORDER = ['Status Quo', 'Innovation', 'Unconstrained']
    
    # 1. US Term Frequency Heatmap 
    # Average US Term freq by Model, Domain, and Condition
    plt.figure(figsize=(14, 8))
    # Filter to AI models only
    hm_df = df[df['model'].isin(['ChatGPT', 'Gemini', 'Sarvam'])]
    
    pivot_us = hm_df.pivot_table(
        values='US_Distinctive_Term_Frequency', 
        index=['model', 'condition'], 
        columns=['domain'], 
        aggfunc='mean'
    ).reindex(COND_ORDER, level='condition')
    
    sns.heatmap(pivot_us, annot=True, cmap="YlOrRd", fmt=".1f", linewidths=.5)
    plt.title("Jurisdictional Drift: US-Distinctive Term Frequency\n(Hits per 1000 words)", fontsize=16, pad=15)
    plt.xlabel("Domain", fontsize=12)
    plt.ylabel("Model & Prompt Condition", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_dir, "us_term_frequency_heatmap.png"), dpi=300)
    print("Saved US Term Frequency Heatmap.")
    
    # 2. GDPR Alignment Heatmap (Data Protection only)
    dp_df = hm_df[hm_df['domain'] == 'Data Protection'].copy()
    if not dp_df.empty:
        plt.figure(figsize=(10, 6))
        pivot_gdpr = dp_df.pivot_table(
            values='GDPR_Mean_Similarity', 
            index=['model', 'condition'], 
            aggfunc='mean'
        ).reindex(COND_ORDER, level='condition')
        
        # Set vmin to compress the color scale around the meaningful range
        # Sarvam-Innovation outlier (0.806) is shown in annotation, 
        # but color scale focuses on 0.88-0.94 range for discrimination
        sns.heatmap(
            pivot_gdpr, 
            annot=True, 
            cmap="Purples", 
            fmt=".3f", 
            linewidths=.5,
            vmin=0.88,
            vmax=0.94
        )
        plt.title("GDPR Alignment: Mean Max Cosine Similarity\n(Data Protection Domain)", fontsize=16, pad=15)
        plt.xlabel("Similarity Score", fontsize=12)
        plt.ylabel("Model & Prompt Condition", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(heatmap_dir, "gdpr_similarity_heatmap.png"), dpi=300)
        print("Saved GDPR Alignment Heatmap.")

if __name__ == "__main__":
    main()
