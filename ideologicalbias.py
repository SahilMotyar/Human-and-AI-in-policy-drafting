import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

COND_ORDER = ['Status Quo', 'Innovation', 'Unconstrained']

FINAL_OUT_DIR = os.path.join("result", "final_outputs")
FIG_DIR = os.path.join(FINAL_OUT_DIR, "figures")
os.makedirs(FINAL_OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

STAGE3_CSV = os.path.join(FINAL_OUT_DIR, "master_dataframe_with_stage3_scores.csv")

BASELINE_PATHS = {
    "Data Protection":  r"extracted_text\core_baselines\Digital Personal Data Protection Act, 2023.txt",
    "Air Pollution":    r"extracted_text\core_baselines\Air (Prevention and Control of Pollution) Act, 1981.txt",
    "National Security":r"extracted_text\core_baselines\National Security Act, 1980.txt"
}

def read_text_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"  [warn] Could not read {filepath}: {e}")
        return ""

# 2. LOAD DATA
print("Loading Stage 3 dataframe...")
if os.path.exists(STAGE3_CSV):
    df = pd.read_csv(STAGE3_CSV)
else:
    print(f"File {STAGE3_CSV} not found! Falling back to master_dataframe.csv if available.")
    df = pd.read_csv("result/archive/master_dataframe.csv")
    
print(f"Loaded {len(df)} records.")

print("Reading full texts for AI drafts...")
# Add full text to dataframe for processing
df['full_text'] = df['file'].apply(lambda fp: read_text_file(fp) if pd.notna(fp) else "")

print("Loading baseline statute texts...")
baseline_texts = [read_text_file(path) for path in BASELINE_PATHS.values()]

def clean_text_for_mcq(text):
    """Remove OCR artifacts before MCQ analysis."""
    # Keep only ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove measurement units and numeric tokens
    text = re.sub(r'\b\d+[\w/]*\b', '', text)
    # Remove very short tokens (OCR noise)
    text = ' '.join(w for w in text.split() if len(w) > 2)
    return text

print("\nComputing Log-Odds Ratio for Ideological Bias...")

ai_corpus_text = clean_text_for_mcq(" ".join(df['full_text'].tolist()))

# Proportionally weight the baseline texts based on the number of AI drafts per domain
baseline_corpus_weighted = []
for domain, path in BASELINE_PATHS.items():
    domain_count = len(df[df['domain'] == domain]) if 'domain' in df.columns else 60
    text = read_text_file(path)
    # Repeat the baseline text to match the AI draft frequency in this domain
    baseline_corpus_weighted.extend([text] * domain_count)

# Apply OCR cleaning when building baseline_corpus_text
baseline_corpus_weighted_clean = [
    clean_text_for_mcq(t) for t in baseline_corpus_weighted
]
baseline_corpus_text = " ".join(baseline_corpus_weighted_clean)

vectorizer = CountVectorizer(stop_words='english', min_df=1)

X = vectorizer.fit_transform([ai_corpus_text, baseline_corpus_text])
vocab = vectorizer.get_feature_names_out()
counts = X.toarray()

ai_counts = counts[0]
baseline_counts = counts[1]

def mcq_log_odds(ai_counts, baseline_counts, prior_counts=None):
    """
    Monroe, Colaresi & Quinn (2008) log-odds ratio.
    """
    if prior_counts is None:
        prior_counts = ai_counts + baseline_counts  # informative prior per word
    
    # Guard against divide-by-zero warnings
    prior_counts = np.maximum(prior_counts, 1)
    
    n_ai   = ai_counts.sum()
    n_base = baseline_counts.sum()
    
    y_ai   = ai_counts   + prior_counts
    y_base = baseline_counts + prior_counts
    
    omega_ai   = y_ai   / (n_ai   + prior_counts.sum())
    omega_base = y_base / (n_base + prior_counts.sum())
    
    log_odds = np.log(omega_ai) - np.log(omega_base)
    variance = (1.0 / y_ai) + (1.0 / y_base)
    z_scores = log_odds / np.sqrt(variance)
    
    return log_odds, z_scores

log_odds, z_scores = mcq_log_odds(ai_counts, baseline_counts)

word_scores = list(zip(vocab, z_scores))
word_scores.sort(key=lambda x: x[1], reverse=True)

print("\n--- Top 15 Words Overrepresented in AI Drafts (Corporate Framing) ---")
corporate_words = word_scores[:15]
for word, word_z in corporate_words:
    print(f"{word}: {word_z:.2f}")

print("\n--- Top 15 Words Overrepresented in Baselines (State Enforcement) ---")
state_words = word_scores[-15:]
for word, word_z in state_words:
    print(f"{word}: {word_z:.2f}")

plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

plot_words = state_words + corporate_words[::-1]  # state at bottom (negative), AI at top (positive)
labels = [w[0] for w in plot_words]
scores = [w[1] for w in plot_words]

colors = ['#d62728' if s < 0 else '#1f77b4' for s in scores]

plt.barh(labels, scores, color=colors)
plt.axvline(0, color='black', linewidth=1.5, linestyle='--')
plt.axvspan(-1.96, 1.96, alpha=0.08, color='gray', label='|z| < 1.96 (n.s.)')
plt.title("Log-Odds Ratio of Lexical Framing\n(AI Drafts vs. Historical Baselines)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Z-Score (Log-Odds Ratio) -> Right is AI Bias, Left is Baseline Bias", fontsize=14, fontweight='bold')
plt.legend(loc='lower right')

plt.tight_layout()
logodds_png = os.path.join(FIG_DIR, "Fig04_Stage4_LogOdds_Framing.png")
plt.savefig(logodds_png, dpi=300)
print(f"\nGenerated Diverging Bar Chart: {logodds_png}")

print("\nScoring texts on Moral Foundations...")

mfd_dict = {
    "Care/Fairness": [
        "rights", "equity", "protect*", "safety", 
        "inclusive", "sustainable", "welfare", "harm*", "benefit*", 
        "justice", "equality", "discrimination", "vulnerable", 
        "public interest", "well-being", "health", "safe-guard*", "consent", "privacy", "mitigat*"
    ],
    "Authority/Loyalty": [
        "imprisonment", "penalty", "cognizable", "sovereignty", "detain*", "punish*", 
        "offence", "jurisdiction", "enforcement", "tribunal", "shall", "liable", "fine", "conviction",
        "state", "national", "security", "mandate*", "compel*", "prohibit*", "compliance", 
        "regulation*", "order", "duty", "obligation", "violation", "lawful", "authorize*", "restrict*"
    ],
    "Technocratic Framing": [
        "stakeholder*", "transparency", "guidelines", "framework", 
        "collaboration", "best practice*"
    ]
}

mfd_regex = {}
for foundation, terms in mfd_dict.items():
    patterns = []
    for term in terms:
        parts = term.split()
        if len(parts) > 1:
            term_str = r'\s+'.join(re.escape(p.replace('*', '')) + (r'\w*' if '*' in p else r'') for p in parts)
            patterns.append(r'\b' + term_str + r'\b')
        elif term.endswith('*'):
            patterns.append(r'\b' + re.escape(term[:-1]) + r'\w*\b')
        else:
            patterns.append(r'\b' + re.escape(term) + r'\b')
    mfd_regex[foundation] = re.compile('|'.join(patterns), re.IGNORECASE)

def score_moral_foundations(text, compiled_regex_dict):
    # Do NOT lowercase before regex — patterns handle case via re.IGNORECASE
    word_count = len(re.findall(r'\b\w+\b', text))
    
    scores = {foundation: 0.0 for foundation in compiled_regex_dict.keys()}
    
    if word_count == 0:
        return scores
        
    for foundation, pattern in compiled_regex_dict.items():
        hits = len(pattern.findall(text))
        scores[foundation] = (hits / word_count) * 1000.0
        
    return scores

print("Processing all drafts for Moral Foundation scores...")
care_scores = []
authority_scores = []
technocratic_scores = []

for idx, row in df.iterrows():
    scores = score_moral_foundations(row['full_text'], mfd_regex)
    care_scores.append(scores['Care/Fairness'])
    authority_scores.append(scores['Authority/Loyalty'])
    technocratic_scores.append(scores['Technocratic Framing'])

# Apply the scoring to every draft in your master dataframe
df['Care_Fairness_Score'] = care_scores
df['Authority_Loyalty_Score'] = authority_scores
df['Technocratic_Framing_Score'] = technocratic_scores

if not baseline_corpus_text.strip() or len(baseline_corpus_text.split()) < 100:
    raise RuntimeError(
        "baseline_corpus_text is empty or too short. "
        "Check BASELINE_PATHS file loading before proceeding."
    )

print("\n--- Per-Condition MCQ Log-Odds (Top 5 per condition) ---")
for cond in COND_ORDER:
    cond_text = clean_text_for_mcq(" ".join(
        df[df['condition'] == cond]['full_text'].tolist()
    ))
    X_cond = vectorizer.transform([cond_text, baseline_corpus_text])
    counts_cond = X_cond.toarray()
    _, z_cond = mcq_log_odds(counts_cond[0], counts_cond[1])

    word_z_cond = sorted(
        zip(vocab, z_cond), key=lambda x: x[1], reverse=True
    )
    top5_ai   = [(w, round(float(z), 2)) for w, z in word_z_cond[:5]]
    top5_base = [(w, round(float(z), 2)) for w, z in word_z_cond[-5:]]
    print(f"\n  Condition: {cond}")
    print(f"    AI-favored (top 5):       {top5_ai}")
    print(f"    Baseline-favored (top 5): {top5_base}")

if 'full_text' in df.columns:
    df = df.drop(columns=['full_text'])

    # --- MFD DISCLAIMER TAGGING ---
    auth_75th = df['Authority_Loyalty_Score'].quantile(0.75)
    care_75th = df['Care_Fairness_Score'].quantile(0.75)

    df['legal_context_risk_authority'] = (df['domain'] == 'National Security') & (df['Authority_Loyalty_Score'] > auth_75th)
    
    # Handle domain name mapping (Data Protection vs DPDP)
    dpdp_mask = df['domain'].isin(['DPDP', 'Data Protection'])
    df['legal_context_risk_care'] = dpdp_mask & (df['Care_Fairness_Score'] > care_75th)
    # ------------------------------

OUTPUT_CSV = os.path.join(FINAL_OUT_DIR, "master_dataframe_with_stage4_scores.csv")
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nStage 4 Complete. Master Dataframe updated with Moral Foundation scores and saved to {OUTPUT_CSV}.")
print("\n--- Checking Cell Sizes for CIs ---")
print(df.groupby(['condition', 'domain'])[['Care_Fairness_Score', 'Authority_Loyalty_Score']].count())

print("\nGenerating National Security Case Study Plot...")

# Calculate NSA baseline scores
ns_baseline_text = read_text_file(BASELINE_PATHS["National Security"])
if not ns_baseline_text.strip():
    raise RuntimeError(
        "National Security baseline text is empty — "
        "check file path in BASELINE_PATHS"
    )
ns_baseline_scores = score_moral_foundations(ns_baseline_text, mfd_regex)
print(f"  NS baseline scores: {ns_baseline_scores}")

ns_df = df[df['domain'] == 'National Security'].copy()

# Melt the dataframe to easily plot both Care and Authority side-by-side
ns_melted = ns_df.melt(id_vars=['condition'], value_vars=['Care_Fairness_Score', 'Authority_Loyalty_Score'], 
                       var_name='Moral Foundation', value_name='Score')
label_map = {
    'Care_Fairness_Score':     'Care/Fairness',
    'Authority_Loyalty_Score': 'Authority/Loyalty'
}
ns_melted['Moral Foundation'] = ns_melted['Moral Foundation'].map(label_map)

plt.figure(figsize=(10, 6))
sns.barplot(data=ns_melted, x='condition', y='Score', hue='Moral Foundation', order=COND_ORDER,
            palette=['#2ca02c', '#d62728'], capsize=.1, err_kws={'linewidth': 1.5}, edgecolor=".2")

# Add horizontal lines for baselines
plt.axhline(ns_baseline_scores['Care/Fairness'], color='#2ca02c', linestyle='--', alpha=0.7, label='Care/Fairness (Baseline)')
plt.axhline(ns_baseline_scores['Authority/Loyalty'], color='#d62728', linestyle='--', alpha=0.7, label='Authority/Loyalty (Baseline)')

plt.title("Moral Foundations Shift in National Security AI Drafts", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Prompt Constraint Level", fontsize=14, fontweight='bold')
plt.ylabel("MFD Score (per 1000 words)", fontsize=14, fontweight='bold')

# Clean up legend to include baselines
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title='Moral Foundation', title_fontsize='13', fontsize='12')
plt.tight_layout()
ns_plot_png = os.path.join(FIG_DIR, "Fig06_Stage4_MFD_NationalSecurity.png")
plt.savefig(ns_plot_png, dpi=300)
print(f"Saved: {ns_plot_png}")

print("Generating Cross-Domain Interaction Plots...")

plt.figure(figsize=(10, 6))
sns.pointplot(data=df, x='condition', y='Authority_Loyalty_Score', hue='domain', order=COND_ORDER, 
              markers=['o', 's', 'D'], linestyles=['-', '--', '-.'], dodge=True, capsize=0.05, err_kws={'linewidth': 1.5})
plt.title("Interaction: Prompt Condition vs Legal Domain\n(Authority/Loyalty Score)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Prompt Constraint Level", fontsize=14, fontweight='bold')
plt.ylabel("Authority/Loyalty Score (per 1000 words)", fontsize=14, fontweight='bold')
plt.legend(title='Legal Domain', title_fontsize='13', fontsize='12')
plt.tight_layout()
auth_plot_png = os.path.join(FIG_DIR, "Fig07_Stage4_Authority_Interaction.png")
plt.savefig(auth_plot_png, dpi=300)
print(f"Saved: {auth_plot_png}")

plt.figure(figsize=(10, 6))
sns.pointplot(data=df, x='condition', y='Care_Fairness_Score', hue='domain', order=COND_ORDER, 
              markers=['o', 's', 'D'], linestyles=['-', '--', '-.'], dodge=True, capsize=0.05, err_kws={'linewidth': 1.5})
plt.title("Interaction: Prompt Condition vs Legal Domain\n(Care/Fairness Score)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Prompt Constraint Level", fontsize=14, fontweight='bold')
plt.ylabel("Care/Fairness Score (per 1000 words)", fontsize=14, fontweight='bold')
plt.legend(title='Legal Domain', title_fontsize='13', fontsize='12')
plt.tight_layout()
care_plot_png = os.path.join(FIG_DIR, "Fig08_Stage4_Care_Interaction.png")
plt.savefig(care_plot_png, dpi=300)
print(f"Saved: {care_plot_png}")

print("Generating Moral Foundations Radar Chart...")

def plot_radar(ax, values, labels, color, label, alpha=0.25):
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values = list(values) + [values[0]]
    ax.plot(angles, values, color=color, linewidth=2, label=label)
    ax.fill(angles, values, color=color, alpha=alpha)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.yaxis.set_tick_params(labelsize=7)

foundations_labels = ['Care/Fairness', 'Authority/Loyalty',
                      'Technocratic\nFraming']
score_cols = ['Care_Fairness_Score', 'Authority_Loyalty_Score',
              'Technocratic_Framing_Score']
radar_colors = {
    'Status Quo':    '#534AB7',
    'Innovation':    '#0F6E56',
    'Unconstrained': '#639922'
}

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Collect raw means for all conditions
all_means_raw = {
    cond: [df[df['condition'] == cond][c].mean() for c in score_cols]
    for cond in radar_colors
}

# Compute baseline means
baseline_scores_radar = []
for foundation in ['Care/Fairness', 'Authority/Loyalty', 'Technocratic Framing']:
    domain_scores = []
    for path in BASELINE_PATHS.values():
        bt = read_text_file(path)
        if bt.strip():
            s = score_moral_foundations(bt, mfd_regex)
            domain_scores.append(s[foundation])
    baseline_scores_radar.append(
        float(np.mean(domain_scores)) if domain_scores else 0.0
    )
all_means_raw['Baseline'] = baseline_scores_radar

# Normalize each dimension to [0,1] across all conditions + baseline
dim_maxes = [
    max(values[i] for values in all_means_raw.values())
    for i in range(len(score_cols))
]
dim_maxes = [m if m > 0 else 1.0 for m in dim_maxes]

all_means = {
    key: [v / dim_maxes[i] for i, v in enumerate(values)]
    for key, values in all_means_raw.items()
}

# Plot normalized values
for cond, col in radar_colors.items():
    plot_radar(ax, all_means[cond], foundations_labels, col, cond)

plot_radar(ax, all_means['Baseline'], foundations_labels,
           '#E24B4A', 'Baseline Statutes', alpha=0.1)

ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
ax.set_title(
    'Moral Foundations Profile (normalized per dimension)\nAI Conditions vs Baseline Statutes',
    fontsize=13, fontweight='bold', pad=25
)

plt.tight_layout()
radar_png = os.path.join(FIG_DIR, 'Fig05_Stage4_MFD_Radar.png')
plt.savefig(radar_png, dpi=300, bbox_inches='tight')
print(f"Saved: {radar_png}")

print("\nAll Stage 4 visualizations successfully generated!")
