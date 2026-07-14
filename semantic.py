import os
import re
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde, spearmanr
from collections import Counter
from bert_score import score, BERTScorer
from transformers import AutoTokenizer
from matplotlib.patches import Ellipse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"BERTScore will run on: {DEVICE}")

MASTER_DF_PATH = "result/archive/master_dataframe.csv"
FINAL_OUT_DIR = os.path.join("result", "final_outputs")
FIG_DIR = os.path.join(FINAL_OUT_DIR, "figures")
os.makedirs(FINAL_OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

BASELINE_DIR = os.path.join("extracted_text", "core_baselines")
BASELINE_PATHS = {
    "Data Protection":   os.path.join(BASELINE_DIR, "Digital Personal Data Protection Act, 2023.txt"),
    "Air Pollution":     os.path.join(BASELINE_DIR, "Air (Prevention and Control of Pollution) Act, 1981.txt"),
    "National Security": os.path.join(BASELINE_DIR, "National Security Act, 1980.txt"),
}

LEGAL_STOPWORDS = {
    'the', 'of', 'and', 'to', 'a', 'in', 'or', 'any', 'be', 'for',
    'such', 'by', 'as', 'an', 'with', 'this', 'that', 'on', 'at', 'is',
    'shall', 'may', 'which', 'it', 'its', 'are', 'been', 'from', 'not'
}

# 2. HELPER FUNCTIONS
def read_text_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

def compute_jsd(text1, text2, remove_stopwords=True):
    tokens1 = re.findall(r"[a-zA-Z']+", text1.lower())
    tokens2 = re.findall(r"[a-zA-Z']+", text2.lower())
    if remove_stopwords:
        tokens1 = [t for t in tokens1 if t not in LEGAL_STOPWORDS]
        tokens2 = [t for t in tokens2 if t not in LEGAL_STOPWORDS]
    if not tokens1 or not tokens2:
        return np.nan
    c1, c2 = Counter(tokens1), Counter(tokens2)
    vocab   = list(set(c1.keys()).union(c2.keys()))
    vec1    = np.array([c1.get(w, 0) for w in vocab], dtype=float)
    vec2    = np.array([c2.get(w, 0) for w in vocab], dtype=float)
    # Add Laplace smoothing to avoid zero-probability terms
    vec1 += 1e-9
    vec2 += 1e-9
    return float(jensenshannon(vec1 / vec1.sum(), vec2 / vec2.sum()))

def chunk_text(text, tokenizer, max_tokens=400, stride=350):
    """Split text into overlapping chunks within the model's token limit."""
    import logging
    # Suppress the token indices > max_length warning from tokenizer
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    chunks = []
    for start in range(0, len(tokens), stride):
        chunk = tokens[start:start + max_tokens]
        if len(chunk) < 50:  # skip tiny trailing chunks
            break
        chunks.append(tokenizer.decode(chunk))
    return chunks if chunks else [text[:1000]]  # fallback

def compute_bert_score_chunked(ai_text, baseline_text, scorer, tokenizer):
    ai_chunks   = chunk_text(ai_text,      tokenizer)
    base_chunks = chunk_text(baseline_text, tokenizer)

    pairs_ai = []
    pairs_base = []
    for ai_chunk in ai_chunks:
        for base_chunk in base_chunks:
            pairs_ai.append(ai_chunk)
            pairs_base.append(base_chunk)

    _, _, f1_all = scorer.score(
        pairs_ai,
        pairs_base,
        batch_size=32
    )

    n_base = len(base_chunks)
    f1_matrix = f1_all.cpu().numpy().reshape(len(ai_chunks), n_base)
    per_chunk_max = f1_matrix.max(axis=1)
    return float(per_chunk_max.mean())

def compute_tfidf_distance(text1, text2):
    vec = TfidfVectorizer(stop_words='english', max_features=5000)
    try:
        X = vec.fit_transform([text1, text2])
        return float(1 - cos_sim(X[0], X[1])[0][0])
    except Exception:
        return np.nan

# 3. DATA LOADING & PREPARATION
print("Loading master dataframe...")
df = pd.read_csv(MASTER_DF_PATH)

df['is_partial'] = df['is_partial'].map(
    lambda x: str(x).strip().lower() in ('true', '1', 'yes')
)
primary_df = df[
    (df['corpus_type'] == 'Primary') &
    (df['is_partial']  == False) &
    (~df['model'].str.contains('sarvam', case=False, na=False))
].copy()
print(f"Filtered: {len(primary_df)} rows — verify this matches expectations")

print("\nCell sizes by condition and model:")
print(primary_df.groupby(['condition', 'model']).size().to_string())
print("\nCell sizes by condition and domain:")
print(primary_df.groupby(['condition', 'domain']).size().to_string())

expected_total = 180
if len(primary_df) != expected_total:
    print(f"[WARN] Expected {expected_total} rows, got {len(primary_df)}")
    print("       Check corpus_type, is_partial, and model filters.")
else:
    print(f"[OK] Row count matches expected: {len(primary_df)}")

print("Loading baseline statutes...")
baselines_text = {domain: read_text_file(path) for domain, path in BASELINE_PATHS.items()}

print("Computing Lexical Drift (JSD)...")
jsd_scores, tfidf_scores, ai_texts, baseline_texts_for_bert = [], [], [], []

for _, row in primary_df.iterrows():
    ai_text       = read_text_file(row['file'])
    baseline_text = baselines_text.get(row['domain'], "")
    ai_texts.append(ai_text)
    baseline_texts_for_bert.append(baseline_text)
    jsd_scores.append(compute_jsd(ai_text, baseline_text))
    tfidf_scores.append(compute_tfidf_distance(ai_text, baseline_text))

primary_df['jsd_score'] = jsd_scores
primary_df['tfidf_distance'] = tfidf_scores

r_val, p_val = spearmanr(primary_df['jsd_score'], primary_df['tfidf_distance'])
print(f"JSD vs TF-IDF convergent validity: r={r_val:.3f}, p={p_val:.4f}")

print("Computing Semantic Drift (BERTScore)...")
MODEL_TYPE = "nlpaueb/legal-bert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_TYPE)
bert_scorer = BERTScorer(model_type=MODEL_TYPE, num_layers=9, device=DEVICE)

CHECKPOINT_PATH = os.path.join(FINAL_OUT_DIR, "bert_scores_checkpoint.csv")

if os.path.exists(CHECKPOINT_PATH):
    checkpoint_df = pd.read_csv(CHECKPOINT_PATH)
    bert_scores_dict = dict(zip(checkpoint_df['file'], checkpoint_df['bert_score_f1']))
    print(f"Resuming from checkpoint: {len(bert_scores_dict)} already scored")
else:
    bert_scores_dict = {}

# Build explicit paired list to avoid index mismatch
scored_rows = list(zip(
    primary_df['file'].tolist(),
    ai_texts,
    baseline_texts_for_bert
))

total = len(scored_rows)
for i, (file_path, ai_text, base_text) in enumerate(scored_rows):
    if file_path not in bert_scores_dict:
        print(f"Scoring ({i+1}/{total}): {os.path.basename(file_path)} on {DEVICE}...")
        
        # --- BERTScore Truncation Guard ---
        ai_tokens = len(tokenizer.encode(ai_text, add_special_tokens=False, truncation=False))
        base_tokens = len(tokenizer.encode(base_text, add_special_tokens=False, truncation=False))
        
        warnings_to_log = []
        if ai_tokens > 512:
            pct_trunc = ((ai_tokens - 512) / ai_tokens) * 100
            warnings_to_log.append(f"{os.path.basename(file_path)} (AI Draft) | Token Count: {ai_tokens} | Truncated: {pct_trunc:.2f}%")
        if base_tokens > 512:
            pct_trunc = ((base_tokens - 512) / base_tokens) * 100
            warnings_to_log.append(f"{os.path.basename(file_path)} (Baseline) | Token Count: {base_tokens} | Truncated: {pct_trunc:.2f}%")
            
        if warnings_to_log:
            with open(os.path.join(FINAL_OUT_DIR, 'truncation_warnings.log'), 'a') as f:
                for w in warnings_to_log:
                    f.write(w + '\n')
        # -----------------------------------
        
        f1 = compute_bert_score_chunked(
            ai_text, base_text, bert_scorer, tokenizer
        )
        bert_scores_dict[file_path] = f1
        print(f"   -> F1: {f1:.4f}")

    if (i + 1) % 5 == 0 or (i + 1) == total:
        pd.DataFrame(
            list(bert_scores_dict.items()),
            columns=['file', 'bert_score_f1']
        ).to_csv(CHECKPOINT_PATH, index=False)
        print(
            f"  BERTScore progress: {i+1}/{total} "
            f"drafts scored (checkpoint saved)"
        )

primary_df['bert_score_f1'] = primary_df['file'].map(bert_scores_dict)

output_csv = os.path.join(FINAL_OUT_DIR, "master_dataframe_with_stage2_scores.csv")
primary_df.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")

print("Generating Visualization...")
def plot_confidence_ellipse(ax, x, y, color, n_std=1.5, alpha=0.15):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        alpha=alpha,
        edgecolor=color,
        linewidth=1.5,
        linestyle='--',
    )
    ax.add_patch(ellipse)

COND_COLORS = {
    'Status Quo':    '#534AB7',
    'Innovation':    '#0F6E56',
    'Unconstrained': '#639922',
}

g = sns.JointGrid(data=primary_df, x='bert_score_f1', y='jsd_score', height=9, ratio=5)
g.plot_joint(sns.scatterplot, hue=primary_df['condition'],
             palette=COND_COLORS, style=primary_df['domain'], s=120, alpha=0.8, edgecolor='white')
g.plot_marginals(sns.kdeplot, hue=primary_df['condition'],
                 palette=COND_COLORS, fill=True, alpha=0.25, linewidth=1.5)

ax = g.ax_joint
x_origin = primary_df['bert_score_f1'].max() + 0.02
y_origin = primary_df['jsd_score'].min() - 0.02

# Add ellipses
for cond, col in COND_COLORS.items():
    sub = primary_df[primary_df['condition'] == cond]
    plot_confidence_ellipse(ax, sub['bert_score_f1'], sub['jsd_score'], col)
    
    # Drift vectors
    mx, my = sub['bert_score_f1'].mean(), sub['jsd_score'].mean()
    ax.annotate('', xy=(mx, my), xytext=(x_origin, y_origin),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.8, alpha=0.6))

# Origin and annotations
ax.scatter(1.0, 0.0, color='#E24B4A', marker='*', s=600, edgecolor='black', label='Baseline Origin (1.0, 0.0)', zorder=5)

ax.annotate(
    'Ideal baseline\norigin (1.0, 0.0)',
    xy=(1.0, 0.0),
    # Center the label further left (was xytext=(0.75, 0.10)) so the text box
    # stays inside the joint axes instead of being clipped by the marginal axis.
    xytext=(0.58, 0.13),
    xycoords='data',
    textcoords='axes fraction',
    ha='center',
    arrowprops=dict(arrowstyle='->', color='#E24B4A', lw=1.5),
    fontsize=9,
    color='#E24B4A',
    bbox=dict(
        boxstyle='round,pad=0.3',
        facecolor='white',
        edgecolor='#E24B4A',
        alpha=0.85
    )
)

r, p = spearmanr(primary_df['bert_score_f1'], primary_df['jsd_score'])
print(f"Spearman r(BERTScore, JSD) = {r:.3f}, p = {p:.4f}")
if r < -0.3 and p < 0.05:
    print("  Confirmed: Higher semantic similarity associates with lower lexical drift")
elif r > 0.3 and p < 0.05:
    print("  Unexpected: Metrics are positively correlated; drift dimensions may be entangled")
else:
    print("  Weak/no monotonic association: JSD and BERTScore may capture independent drift axes")

print("\nPer-condition Spearman r(BERTScore, JSD):")
for cond in primary_df['condition'].unique():
    sub = primary_df[primary_df['condition'] == cond]
    if len(sub) > 5:
        r_c, p_c = spearmanr(
            sub['bert_score_f1'], sub['jsd_score']
        )
        print(f"  {cond}: r={r_c:.3f}, p={p_c:.4f}, n={len(sub)}")
ax.text(0.02, 0.97, f"Spearman r = {r:.3f} (p={p:.3f})", transform=ax.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax.set_xlim(primary_df['bert_score_f1'].min() - 0.05, 1.05)
ax.set_ylim(-0.05, primary_df['jsd_score'].max() + 0.05)

plt.suptitle('Stage 2 - Algorithmic Drift: Baseline vs Semantic Drift', fontsize=16, y=1.02)
ax.set_xlabel('Semantic Similarity (BERTScore F1) $\\rightarrow$', fontsize=12)
ax.set_ylabel('Lexical Drift (Jensen-Shannon Divergence) $\\rightarrow$', fontsize=12)

joint_legend = ax.get_legend()
if joint_legend is not None:
    handles, labels = ax.get_legend_handles_labels()
    joint_legend.remove()
    g.fig.legend(
        handles,
        labels,
        bbox_to_anchor=(1.15, 0.85),
        loc='upper right',
        fontsize=9,
        framealpha=0.9,
    )
ax.grid(True, linestyle='--', alpha=0.6)

stage2_fig = os.path.join(FIG_DIR, 'Fig01_Stage2_Semantic_Lexical_Drift.png')
g.savefig(stage2_fig, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{stage2_fig}'")

summary = primary_df.groupby('condition').agg(
    n             = ('bert_score_f1', 'count'),
    bert_mean     = ('bert_score_f1', 'mean'),
    bert_std      = ('bert_score_f1', 'std'),
    jsd_mean      = ('jsd_score',     'mean'),
    jsd_std       = ('jsd_score',     'std'),
    tfidf_mean    = ('tfidf_distance','mean'),
    tfidf_std     = ('tfidf_distance','std'),
).round(4)
print("\nCondition Summary Table:")
print(summary.to_string())
summary.to_csv(os.path.join(FINAL_OUT_DIR, "stage2_condition_summary.csv"))

# Also save domain-level summary
domain_summary = primary_df.groupby(['domain', 'condition']).agg(
    n          = ('bert_score_f1', 'count'),
    bert_mean  = ('bert_score_f1', 'mean'),
    jsd_mean   = ('jsd_score',     'mean'),
    tfidf_mean = ('tfidf_distance','mean'),
).round(4)
print("\nDomain x Condition Summary:")
print(domain_summary.to_string())
domain_summary.to_csv(os.path.join(FINAL_OUT_DIR, "stage2_domain_condition_summary.csv"))