import os
import sys
import subprocess
import warnings
import re

def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

for pkg in ['pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'spacy']:
    install_if_missing(pkg)

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[WARN] spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

warnings.filterwarnings('ignore')

# --- CONSTANTS & PATHS ---
BASE_DIR = r"."
INPUT_CSV = os.path.join(BASE_DIR, "result", "final_outputs", "master_dataframe_with_stage5_scores.csv")
INPUT_CSV4 = os.path.join(BASE_DIR, "result", "final_outputs", "master_dataframe_with_stage4_scores.csv")
FINAL_OUT_DIR = os.path.join(BASE_DIR, "result", "final_outputs")
FIG_DIR = os.path.join(FINAL_OUT_DIR, "figures")
OUT_DIR = FINAL_OUT_DIR
BASELINE_DIR = os.path.join(BASE_DIR, "extracted_text", "core_baselines")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Hedge phrases for complexity compute
HEDGE_PHRASES = [
    r'provided that', r'provided further that', r'provided also that',
    r'except where', r'except that', r'save as', r'save as otherwise provided',
    r'unless otherwise', r'unless the context otherwise requires',
    r'notwithstanding', r'notwithstanding anything contained', r'subject to',
    r'subject to the provisions', r'without prejudice', r'without prejudice to the generality',
    r'where applicable', r'as the case may be', r'in so far as', r'to the extent',
    r'to such extent', r'nothing in this', r'nothing herein', r'shall not apply',
    r'for the removal of doubts', r'mutatis mutandis', r'may be prescribed',
    r'as may be prescribed', r'in accordance with', r'pursuant to', r'having regard to',
    r'as it may deem fit', r'as they may deem fit', r'on such terms and conditions'
]

HEDGE_PATTERN = re.compile(r'\b(?:' + '|'.join(HEDGE_PHRASES) + r')\b', re.IGNORECASE)
MATTR_WINDOW = 50

# --- METRIC FUNCTIONS ---
def compute_mattr(text, window=MATTR_WINDOW):
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if len(tokens) < window * 2:
        return np.nan
    ttrs = [
        len(set(tokens[i:i + window])) / window
        for i in range(len(tokens) - window + 1)
    ]
    return float(np.mean(ttrs))

def compute_hedge_density(text):
    word_count = len(re.findall(r'\S+', text))
    if word_count == 0:
        return np.nan
    hits = len(HEDGE_PATTERN.findall(text))
    return (hits / word_count) * 100.0

def split_sentences(text):
    if nlp is None:
        text = re.sub(r'\b(Sec\.|Cl\.|s\.|Art\.|No\.|vs\.)\s+', r'\1_', text)
        sentences = re.split(r'(?<=[.!?;])\s+(?=[A-Z(\"\d])', text.strip())
        return [s for s in sentences if len(s.split()) >= 3]
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.split()) >= 3]

def compute_sentence_stats(text):
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return np.nan, np.nan
    lengths = [len(s.split()) for s in sentences]
    return float(np.mean(lengths)), float(np.var(lengths))

# --- MAIN EXECUTION ---
print("Loading Master Data...")
df = pd.read_csv(INPUT_CSV)

# Identify valid Sarvam rows (exclude 'refusal_detected' or 'under_3KB')
sarvam_mask = (df['model'] == 'Sarvam') & \
              (~df['issues'].fillna('').str.contains(r'\b(refusal_detected|under_3KB)\b', case=False, regex=True))

valid_sarvam_count = sarvam_mask.sum()
print(f"Executing inner calculation... Valid Sarvam rows to process (complexity patching): {valid_sarvam_count}")

# COMPUTE SARVAM COMPLEXITIES ON THE FLY
domain_baselines = {
    'Data Protection': os.path.join(BASELINE_DIR, 'Digital Personal Data Protection Act, 2023.txt'),
    'Air Pollution': os.path.join(BASELINE_DIR, 'Air (Prevention and Control of Pollution) Act, 1981.txt'),
    'National Security': os.path.join(BASELINE_DIR, 'National Security Act, 1980.txt')
}

baseline_vars = {}
baseline_mattr = []
baseline_hedge = []
baseline_mean_sl = []
baseline_sl_var = []

for domain, path in domain_baselines.items():
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        msl, slv = compute_sentence_stats(text)
        baseline_vars[domain] = slv
        
        baseline_mattr.append(compute_mattr(text))
        baseline_hedge.append(compute_hedge_density(text))
        baseline_mean_sl.append(msl)
        baseline_sl_var.append(slv)

grand_baseline_means = {
    'mattr': np.mean(baseline_mattr) if baseline_mattr else 0,
    'hedge_density': np.mean(baseline_hedge) if baseline_hedge else 0,
    'mean_sent_len': np.mean(baseline_mean_sl) if baseline_mean_sl else 0,
    'sent_len_var': np.mean(baseline_sl_var) if baseline_sl_var else 0
}

# NOTE ON sent_len_var COLLECTION METHOD CONFOUND:
# Sarvam was accessed via API (max_token setting) producing ~1900 word docs.
# ChatGPT and Gemini were accessed via incognito web UI producing ~850-1077 word docs.
# sent_len_var correlates with document length (more sentences = more variance opportunity).
# This means sent_len_var comparisons between Sarvam and Western models are partially
# confounded by the collection method, not purely intrinsic model behaviour.
# MATTR, US_Distinctive_Term_Frequency, and GDPR_Mean_Similarity are length-normalised
# and should be used as primary evidence for sovereign AI differences.
# sent_len_var is reported as indicative only.

processed = 0
for idx in df[sarvam_mask].index:
    file_path = df.loc[idx, 'file']
    abs_path = file_path if os.path.isabs(file_path) else os.path.join(BASE_DIR, file_path)
    
    if os.path.exists(abs_path):
        with open(abs_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        mattr = compute_mattr(text)
        hedge_density = compute_hedge_density(text)
        mean_sent_len, sent_len_var = compute_sentence_stats(text)
        
        domain = df.loc[idx, 'domain']
        b_var = baseline_vars.get(domain, None)
        sent_var_red = ((sent_len_var - b_var) / b_var) * 100 if b_var else np.nan
        
        df.loc[idx, 'mattr'] = mattr
        df.loc[idx, 'hedge_density'] = hedge_density
        df.loc[idx, 'mean_sent_len'] = mean_sent_len
        df.loc[idx, 'sent_len_var'] = sent_len_var
        df.loc[idx, 'sent_var_red'] = sent_var_red
        
        processed += 1

# Save computed df back directly
try:
    df.to_csv(INPUT_CSV, index=False)
except Exception as e:
    print(f"Warning: Could not save overwritten metadata to DB {e}")

sarvam_df = df[df['model'] == 'Sarvam'].copy()
western_df = df[df['model'].isin(['ChatGPT', 'Gemini'])].copy()

# === PART A: Data Loading and Verification ===
print("\n=== PART A: Data Loading and Verification ===")
print("Sarvam Data Inventory:")
print(sarvam_df.groupby(['domain', 'condition']).size())
print(f"Total Sarvam drafts: {len(sarvam_df)}")
print(f"Complete domains (at least 1 row): {sarvam_df.groupby('domain')['condition'].nunique().to_dict()}\n")

def get_word_count(filepath):
    try:
        abs_path = filepath if os.path.isabs(filepath) else os.path.join(BASE_DIR, filepath)
        with open(abs_path, 'r', encoding='utf-8') as f:
            return len(f.read().split())
    except:
        return 0

sarvam_df['word_count'] = sarvam_df['file'].apply(get_word_count)
sarvam_df['is_short'] = sarvam_df['word_count'] < 500

print("Sarvam National Security — short/refusal outputs:")
ns_sarvam = sarvam_df[sarvam_df['domain'] == 'National Security']
print(ns_sarvam[['condition', 'word_count', 'is_short']].to_string())

# === PART B: Refusal Rate Analysis ===
refusal_stats = []
for cond in ['Status Quo', 'Innovation', 'Unconstrained']:
    cond_df = ns_sarvam[ns_sarvam['condition'] == cond]
    total = len(cond_df)
    short = cond_df['is_short'].sum()
    complete = total - short
    rate = (short / total) * 100 if total > 0 else 0
    refusal_stats.append({
        'Condition': cond, 'Total': total, 'Short (<500w)': short, 
        'Complete': complete, 'Refusal Rate': f"{rate:.1f}%"
    })
    
refusal_df = pd.DataFrame(refusal_stats)
print("\nNational Security Refusal Rates (Sarvam):")
print(refusal_df.to_markdown(index=False))
refusal_df.to_csv(os.path.join(OUT_DIR, 'national_security_refusal_analysis.csv'), index=False)

# === PART C: Metric Comparison ===
COMP_METRICS = [
    'mattr', 'hedge_density', 'mean_sent_len', 'sent_len_var',
    'Care_Fairness_Score', 'Authority_Loyalty_Score', 
    'Technocratic_Framing_Score', 'US_Distinctive_Term_Frequency',
    'GDPR_Mean_Similarity'
]

comp_results = []
summary_results = []

for domain in ['Air Pollution', 'Data Protection']:
    s_dom = sarvam_df[(sarvam_df['domain'] == domain) & (~sarvam_df['issues'].fillna('').str.contains(r'\b(refusal_detected|under_3KB)\b', case=False, regex=True))]
    w_dom = western_df[western_df['domain'] == domain]
    
    for metric in COMP_METRICS:
        s_vals = s_dom[metric].dropna().values
        w_vals = w_dom[metric].dropna().values
        
        if len(s_vals) > 0 and len(w_vals) > 0:
            s_mean, s_std = np.mean(s_vals), np.std(s_vals)
            w_mean, w_std = np.mean(w_vals), np.std(w_vals)
            u_stat, p_val = stats.mannwhitneyu(s_vals, w_vals, alternative='two-sided')
            
            n1, n2 = len(s_vals), len(w_vals)
            r_eff = 1 - (2 * u_stat) / (n1 * n2) if n1*n2 > 0 else np.nan
            
            comp_results.append({
                'Metric': metric, 'Domain': domain, 
                'Sarvam_mean': s_mean, 'Sarvam_std': s_std,
                'Western_mean': w_mean, 'Western_std': w_std,
                'U_stat': u_stat, 'p_value': p_val, 'effect_size_r': r_eff
            })
            
            diff = s_mean - w_mean
            pct_diff = (diff / w_mean) * 100 if w_mean != 0 else np.nan
            summary_results.append({
                'Metric': metric, 'Domain': domain,
                'Sarvam_mean': round(s_mean, 4), 'Western_mean': round(w_mean, 4),
                'Difference': round(diff, 4), '%_Difference': round(pct_diff, 2),
                'MannWhitney_p': round(p_val, 4), 'Significant': 'Yes' if p_val < 0.05 else 'No'
            })

pd.DataFrame(comp_results).to_csv(os.path.join(OUT_DIR, 'sarvam_vs_western_metrics.csv'), index=False)
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(os.path.join(OUT_DIR, 'sarvam_summary_table.csv'), index=False)
print("\n=== Sarvam Summary Table ===")
print(summary_df.to_markdown(index=False))

# === PART D: Complexity Profile Visualization ===")
print("\n=== PART D: Complexity Profile Visualization ===")
complex_metrics = ['mattr', 'hedge_density', 'mean_sent_len', 'sent_len_var']

# Filters for plot
cg_mask = (df['corpus_type'] == 'Primary') & \
          (df['model'].isin(['ChatGPT', 'Gemini'])) & \
          (df['domain'].isin(['Air Pollution', 'Data Protection']))
sarvam_plot_mask = (df['corpus_type'] == 'Supplementary') & \
                   (df['model'] == 'Sarvam') & \
                   (~df['issues'].fillna('').str.contains(r'\b(refusal_detected|under_3KB)\b', case=False, regex=True)) & \
                   (df['domain'].isin(['Air Pollution', 'Data Protection']))
    
plot_df = pd.concat([df[sarvam_plot_mask], df[cg_mask]], ignore_index=True)
plot_df['model'] = pd.Categorical(plot_df['model'], categories=['Sarvam', 'ChatGPT', 'Gemini'], ordered=True)

palette = {'ChatGPT': '#534AB7', 'Gemini': '#0F6E56', 'Sarvam': '#C84B31'}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Sovereign vs Western Model: Complexity Profile Comparison\n(Air Pollution + Data Protection — Sarvam complete drafts only)", fontsize=16)

out_plot_path = os.path.join(FIG_DIR, "Fig10_Stage8_Sarvam_Complexity.png")

for ax, metric in zip(axes.flatten(), complex_metrics):
    sns.boxplot(data=plot_df, x='model', y=metric, hue='model', palette=palette, order=['Sarvam', 'ChatGPT', 'Gemini'], ax=ax, legend=False)
    ax.set_title(metric)
    ax.set_xlabel('')
        
    # Add grand baseline mean line
    if metric in grand_baseline_means:
        ax.axhline(grand_baseline_means[metric], color='red', linestyle='--', label='Grand Baseline Mean')
        
    # Add 'n' labels above boxes
    for i, model in enumerate(['Sarvam', 'ChatGPT', 'Gemini']):
        model_data = plot_df[plot_df['model'] == model][metric].dropna()
        n = len(model_data)
        if n > 0:
            max_val = model_data.max()
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            ax.text(i, max_val + (y_range * 0.02), f'n={n}', horizontalalignment='center', verticalalignment='bottom', size='medium', color='black', weight='semibold')

fig.text(
    0.5,
    0.01,
    "Sarvam National Security excluded - 70% short/incomplete output rate under Unconstrained condition "
    "(3/10 confirmed refusals, 7/10 short outputs).\n"
    "Sarvam Air+DP n=51 (52 raw rows; 1 excluded due to under_3KB/refusal_detected).",
    ha='center',
    fontsize=10
)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(out_plot_path, dpi=300)
plt.close()
print(f"Generated Plot D: {out_plot_path}")


# === PART E: Jurisdictional Drift Comparison ===
print("\n=== PART E: Jurisdictional Drift Comparison ===")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Jurisdictional Drift: Sovereign vs Western Models")

# Re-filter overall valid rows for drift calculations
plot_df_drift = pd.concat([df[sarvam_plot_mask], df[cg_mask]], ignore_index=True)
plot_df_drift['model'] = pd.Categorical(plot_df_drift['model'], categories=['Sarvam', 'ChatGPT', 'Gemini'], ordered=True)

drift_df = plot_df_drift[plot_df_drift['domain'].isin(['Air Pollution', 'Data Protection'])]
us_term_means = (
    drift_df.groupby(['domain', 'model'], as_index=False)['US_Distinctive_Term_Frequency']
    .mean()
)

sns.barplot(data=us_term_means,
            x='domain', y='US_Distinctive_Term_Frequency', hue='model',
            palette=palette, hue_order=['Sarvam', 'ChatGPT', 'Gemini'], ax=ax1, errorbar=None)
ax1.set_title("US Term Frequency by Model and Domain\n(National Security excluded - Sarvam 70% short/incomplete output rate)")

dp_df = plot_df_drift[plot_df_drift['domain'] == 'Data Protection']
sns.barplot(data=dp_df, x='condition', y='GDPR_Mean_Similarity', hue='model',
            palette=palette, hue_order=['Sarvam', 'ChatGPT', 'Gemini'], ax=ax2, errorbar='se')
ax2.set_title("GDPR Similarity (Data Protection Only)")
western_gdpr_mean = dp_df[dp_df['model'].isin(['ChatGPT', 'Gemini'])]['GDPR_Mean_Similarity'].mean()
ax2.axhline(western_gdpr_mean, color='red', linestyle='--', label='Western Mean')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'Fig11_Stage8_Sarvam_Jurisdictional.png'), dpi=300)
plt.close()

print("\nAnalysis Complete. All integrated Sarvam analysis and dynamic metric overrides saved to result/final_outputs/")
