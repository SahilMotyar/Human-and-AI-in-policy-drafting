import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from collections import Counter
from scipy import stats
from scipy.stats import kruskal, spearmanr, mannwhitneyu

import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[WARN] spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

try:
    from scikit_posthocs import posthoc_dunn
    POSTHOC_AVAILABLE = True
except ImportError:
    posthoc_dunn = None
    POSTHOC_AVAILABLE = False
    print("[WARN] scikit-posthocs not installed. Run: pip install scikit-posthocs")
    print("       Dunn's post-hoc tests will be skipped.")

STAGE2_CSV = os.path.join("result", "final_outputs", "master_dataframe_with_stage2_scores.csv")   # output from Stage 2
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

MATTR_WINDOW = 50   # sliding window size for MATTR
COND_ORDER = ['Status Quo', 'Innovation', 'Unconstrained']

HEDGE_PHRASES = [
    # Core Provisos & Exceptions
    r'provided that',
    r'provided further that',
    r'provided also that',
    r'except where',
    r'except that',
    r'save as',
    r'save as otherwise provided',
    r'unless otherwise',
    r'unless the context otherwise requires',
    
    # Non-Obstante & Overriding Clauses
    r'notwithstanding',
    r'notwithstanding anything contained',
    r'subject to',
    r'subject to the provisions',
    r'without prejudice',
    r'without prejudice to the generality',
    
    # Conditionals & Qualifications
    r'where applicable',
    r'as the case may be',
    r'in so far as',
    r'to the extent',
    r'to such extent',
    r'nothing in this',
    r'nothing herein',
    r'shall not apply',
    r'for the removal of doubts',
    r'mutatis mutandis',
    
    # Delegation & Discretion Hedges
    r'may be prescribed',
    r'as may be prescribed',
    r'in accordance with',
    r'pursuant to',
    r'having regard to',
    r'as it may deem fit',
    r'as they may deem fit',
    r'on such terms and conditions'
]
HEDGE_PATTERN = re.compile(r'\b(?:' + '|'.join(HEDGE_PHRASES) + r')\b', re.IGNORECASE)

# 2. HELPER FUNCTIONS
def read_text_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"  [warn] Could not read {filepath}: {e}")
        return ""

def compute_mattr(text, window=MATTR_WINDOW):
    """
    Measure of Textual Lexical Diversity (MATTR).
    Slides a fixed window across the token sequence and averages the
    type-token ratio at each step, eliminating the length bias of plain TTR.
    Range 0–1; higher = more lexically diverse per window.
    NOTE: Real Indian statutes score ~0.62–0.66 due to heavy repetition of
    defined terms. AI drafts typically score higher (0.72–0.84).
    """
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if len(tokens) < window * 2:
        return np.nan
    ttrs = [
        len(set(tokens[i:i + window])) / window
        for i in range(len(tokens) - window + 1)
    ]
    return float(np.mean(ttrs))

def compute_hedge_density(text):
    """
    Legal Hedge Density: count of enforcement conjunction phrases per 100 words.
    Higher = more conditional legal architecture preserved.
    Typical Indian statute sections: ~0.6–1.2 per 100 words.
    AI drafts: ~0.1–0.8 per 100 words depending on prompt condition.
    """
    word_count = len(re.findall(r'\S+', text))
    if word_count == 0:
        return np.nan
    hits = len(HEDGE_PATTERN.findall(text))
    return (hits / word_count) * 100.0

def split_sentences(text):
    """
    spaCy-based sentence splitter for robust handling of legal texts.
    Filters out fragments under 3 words.
    """
    if nlp is None:
        # Fallback regex if spacy not available
        text = re.sub(r'\b(Sec\.|Cl\.|s\.|Art\.|No\.|vs\.)\s+', r'\1_', text)
        sentences = re.split(r'(?<=[.!?;])\s+(?=[A-Z(\"\d])', text.strip())
        return [s for s in sentences if len(s.split()) >= 3]
    
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.split()) >= 3]

def compute_sentence_stats(text):
    """
    Returns (mean_sentence_length, sentence_length_variance).
    High variance = complex nested structures (short definitional + long operative).
    Low variance = syntactic uniformity (key compression signal).
    Indian statutes: mean ~35–50w, variance ~400–900.
    AI drafts: mean ~18–40w, variance ~80–1500 depending on condition.
    """
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return np.nan, np.nan
    lengths = [len(s.split()) for s in sentences]
    return float(np.mean(lengths)), float(np.var(lengths))

# 3. LOAD DATA
print("Loading Stage 2 dataframe...")
df = pd.read_csv(STAGE2_CSV)
print(f"  {len(df)} drafts loaded.")

print("Loading baseline statute texts...")
baselines = {domain: read_text_file(path) for domain, path in BASELINE_PATHS.items()}

print("Computing Stage 3 metrics on all AI drafts...")

mattr_vals, hedge_vals, mean_sl_vals, var_sl_vals = [], [], [], []

for seq_idx, (idx, row) in enumerate(df.iterrows()):
    text = read_text_file(row['file'])
    mattr_vals.append(compute_mattr(text))
    hedge_vals.append(compute_hedge_density(text))
    mean_sl, var_sl = compute_sentence_stats(text)
    mean_sl_vals.append(mean_sl)
    var_sl_vals.append(var_sl)
    if (seq_idx + 1) % 30 == 0:
        print(f"  Processed {seq_idx + 1}/{len(df)} drafts...")

df['mattr']         = mattr_vals
df['hedge_density'] = hedge_vals
df['mean_sent_len'] = mean_sl_vals
df['sent_len_var']  = var_sl_vals

print("\n--- MATTR Window Sensitivity Analysis ---")
for window in [30, 50, 100]:
    col = f'mattr_w{window}'
    df[col] = df['file'].apply(
        lambda f, w=window: compute_mattr(read_text_file(f), window=w)
    )

_mattr_corr = df[['mattr_w30', 'mattr_w50', 'mattr_w100']].dropna()
if len(_mattr_corr) > 1:
    r_30_50, _ = spearmanr(_mattr_corr['mattr_w30'], _mattr_corr['mattr_w50'])
    r_50_100, _ = spearmanr(_mattr_corr['mattr_w50'], _mattr_corr['mattr_w100'])
else:
    r_30_50, r_50_100 = np.nan, np.nan

print(f"  r(w30, w50)  = {r_30_50:.3f}")
print(f"  r(w50, w100) = {r_50_100:.3f}")
if pd.notna(r_30_50) and pd.notna(r_50_100) and min(r_30_50, r_50_100) > 0.90:
    print("  Window choice is stable — results robust to window size")
else:
    print("  [warn] MATTR is window-sensitive — report all three in paper")

# Drop the temporary columns after analysis
df.drop(columns=['mattr_w30', 'mattr_w50', 'mattr_w100'], inplace=True)

print("Computing baseline statute metrics...")
baseline_stats = {}
for domain, text in baselines.items():
    mean_sl, var_sl = compute_sentence_stats(text)
    baseline_stats[domain] = {
        'mattr':         compute_mattr(text),
        'hedge_density': compute_hedge_density(text),
        'mean_sent_len': mean_sl,
        'sent_len_var':  var_sl,
    }
    m = baseline_stats[domain]
    print(f"  {domain}: MATTR={m['mattr']:.3f}  Hedge={m['hedge_density']:.3f}  "
          f"MeanSL={m['mean_sent_len']:.1f}  VarSL={m['sent_len_var']:.1f}")

BASELINE_GRAND = {
    metric: float(np.mean([baseline_stats[d][metric] for d in baseline_stats]))
    for metric in ['mattr', 'hedge_density', 'mean_sent_len', 'sent_len_var']
}
print("\nGrand baseline means used for reference lines:")
for k, v in BASELINE_GRAND.items():
    print(f"  {k}: {v:.3f}")

print("\n--- Per-Statute Baseline Metrics (verify comparability before averaging) ---")
print(f"{'Domain':<20} {'MATTR':>8} {'Hedge':>8} {'MeanSL':>8} {'VarSL':>8}")
print("-" * 56)
for domain, m in baseline_stats.items():
    print(f"{domain:<20} {m['mattr']:>8.3f} {m['hedge_density']:>8.3f} "
        f"{m['mean_sent_len']:>8.1f} {m['sent_len_var']:>8.1f}")
print(f"{'Grand Mean':<20} "
    f"{BASELINE_GRAND['mattr']:>8.3f} "
    f"{BASELINE_GRAND['hedge_density']:>8.3f} "
    f"{BASELINE_GRAND['mean_sent_len']:>8.1f} "
    f"{BASELINE_GRAND['sent_len_var']:>8.1f}")

def rank_biserial(a, b):
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0: return np.nan
    u, _ = mannwhitneyu(a, b, alternative='two-sided')
    return 1 - (2 * u) / (n1 * n2)

df['sent_var_red'] = df.apply(
    lambda r: (
        ((r['sent_len_var'] - baseline_stats[r['domain']]['sent_len_var']) 
         / baseline_stats[r['domain']]['sent_len_var']) * 100
        if r['domain'] in baseline_stats and pd.notna(r['sent_len_var'])
           and baseline_stats[r['domain']]['sent_len_var'] > 0
        else np.nan
    ), axis=1
)

print("\n--- Kruskal-Wallis Statistics ---")
for metric in ['mattr', 'hedge_density', 'mean_sent_len', 'sent_len_var']:
    groups = [df[df['condition'] == c][metric].dropna() for c in COND_ORDER]
    if all(len(g) > 0 for g in groups):
        stat, p = kruskal(*groups)
        n_total = sum(len(g) for g in groups)
        k = len(groups)
        eta2 = (stat - k + 1) / (n_total - k)
        print(f"{metric}: H={stat:.3f}, p={p:.4f}, η²={eta2:.3f}")
        if p < 0.05 and POSTHOC_AVAILABLE:
            try:
                # Perform Dunn's post-hoc test
                dunn_res = posthoc_dunn(df, val_col=metric, group_col='condition', p_adjust='bonferroni')
                print("  Dunn's post-hoc (Bonferroni-adjusted):")
                print(dunn_res)
            except Exception as e:
                print(f"  [Dunn's test error]: {e}")
print("---------------------------------\n")

print("\n--- Rank-Biserial Correlation Effect Sizes (each condition vs grand mean) ---")
for metric in ['mattr', 'hedge_density', 'mean_sent_len', 'sent_len_var']:
    print(f"\n{metric}:")
    groups = {c: df[df['condition'] == c][metric].dropna().values
              for c in COND_ORDER}
    for c1, c2 in [('Status Quo', 'Unconstrained'),
                   ('Status Quo', 'Innovation'),
                   ('Innovation', 'Unconstrained')]:
        rb = rank_biserial(groups[c1], groups[c2])
        magnitude = ('negligible' if abs(rb) < 0.1 else
                     'small'      if abs(rb) < 0.3 else
                     'medium'     if abs(rb) < 0.5 else 'large')
        print(f"  {c1} vs {c2}: r_rb={rb:.3f} ({magnitude})")

print("\n--- Per-Model Kruskal-Wallis ---")
for model in df['model'].unique():
    model_df = df[df['model'] == model]
    print(f"\n  Model: {model}")
    for metric in ['mattr', 'hedge_density', 'mean_sent_len', 'sent_len_var']:
        groups = [model_df[model_df['condition'] == c][metric].dropna()
                  for c in COND_ORDER]
        if all(len(g) > 2 for g in groups):
            stat, p = kruskal(*groups)
            print(f"    {metric}: H={stat:.3f}, p={p:.4f}")

# Save augmented dataframe
output_csv = os.path.join(FINAL_OUT_DIR, "master_dataframe_with_stage3_scores.csv")
df.to_csv(output_csv, index=False)
print(f"\nSaved metrics to {output_csv}")

print("Generating Stage 3 visualization...")

COND_COLORS = {
    'Status Quo':    '#534AB7',
    'Innovation':    '#0F6E56',
    'Unconstrained': '#639922',
}
MODEL_ORDER = ['ChatGPT', 'Gemini']

# Describe whether the baseline sits ABOVE or BELOW the AI data cloud.
BASELINE_ABOVE_DATA = {
    'mattr':          False,   # statute MATTR < AI drafts (AI is more lexically diverse)
    'hedge_density':  True,    # statute hedge density > AI drafts (AI strips hedges)
    'mean_sent_len':  True,    # statute sentences longer than AI drafts
    'sent_len_var':   True,    # statute variance >= AI drafts (except Status Quo)
}

METRICS = [
    ('mattr',
     'MATTR  (Lexical Diversity)',
     'MATTR score',
     'Higher = more lexically diverse per 50-word window',
     'Statute MATTR lower than AI\n(statutes repeat defined terms heavily)',
     ),
    ('hedge_density',
     'Hedge Density  (Conjunctions / 100 words)',
     'Hedges per 100 words',
     'Higher = more conditional legal architecture preserved',
     'AI drafts fall short of statute\nhedge density across all conditions',
     ),
    ('mean_sent_len',
     'Mean Sentence Length  (words)',
     'Words per sentence',
     'Higher = longer, more nested operative clauses',
     'AI sentences shorter than statute\n(operative clause compression)',
     ),
    ('sent_len_var',
     'Sentence Length Variance',
     'Variance  (words²)',
     'Higher = wider mix of short definitional + long operative sentences',
     'Status Quo closest to statute variance;\nUnconstrained collapses toward uniformity',
     ),
]

BOX_W      = 0.55
WITHIN_GAP = 0.72
GROUP_GAP  = 1.30

fig, axes = plt.subplots(1, 4, figsize=(22, 9), facecolor='white')
# top lowered (was 0.87) to give the raised panel titles headroom below the suptitle
fig.subplots_adjust(left=0.05, right=0.97, top=0.84, bottom=0.22, wspace=0.42)

for ax, (metric, title, ylabel, subtitle, finding_note) in zip(axes, METRICS):

    positions = []
    model_centres = []
    offset = 0.0
    for model in MODEL_ORDER:
        cluster = []
        for cond in COND_ORDER:
            pos = offset + len(cluster) * WITHIN_GAP
            cluster.append(pos)
            positions.append((model, cond, pos))
        model_centres.append(float(np.mean(cluster)))
        offset = cluster[-1] + GROUP_GAP + WITHIN_GAP

    for model, cond, pos in positions:
        vals = df[(df['model'] == model) & (df['condition'] == cond)][metric].dropna().values
        col  = COND_COLORS[cond]
        ax.boxplot(
            vals, positions=[pos], widths=BOX_W,
            patch_artist=True, notch=False, showfliers=True,
            whiskerprops=dict(color=col, lw=1.5),
            capprops    =dict(color=col, lw=2.0),
            medianprops =dict(color='white', lw=2.5),
            flierprops  =dict(marker='o', markerfacecolor=col,
                              markersize=4, alpha=0.50, markeredgewidth=0),
            boxprops    =dict(facecolor=col, alpha=0.72, edgecolor=col, lw=1.0),
        )

    # Dynamic y-limits — accommodate both data and baseline line with headroom
    all_vals = df[metric].dropna().values
    bval     = BASELINE_GRAND[metric]
    y_lo = min(np.percentile(all_vals, 1),  bval)
    y_hi = max(np.percentile(all_vals, 99), bval)
    span = y_hi - y_lo
    ax.set_ylim(y_lo - span * 0.10, y_hi + span * 0.32)

    ax.axhline(bval, color='#E24B4A', lw=1.8, ls='--', alpha=0.88, zorder=0)

    x_positions = [p for _, _, p in positions]
    q75 = np.percentile(all_vals, 75)
    q25 = np.percentile(all_vals, 25)
    if BASELINE_ABOVE_DATA[metric]:
        label_y = max(bval + span * 0.04, q75 + span * 0.06)
    else:
        label_y = min(bval - span * 0.04, q25 - span * 0.06)
    
    ax.text(
        np.mean(x_positions), label_y,
        f"Statute ref: {bval:.2f}\n{finding_note}",
        ha='center', va='bottom', fontsize=7.0, color='#c0392b',
        linespacing=1.35,
        bbox=dict(facecolor='white', alpha=0.82,
                  edgecolor='#f0a0a0', boxstyle='round,pad=0.3', lw=0.6)
    )

    ax_ylim = ax.get_ylim()
    y_span  = ax_ylim[1] - ax_ylim[0]

    for model, cond, pos in positions:
        vals = df[(df['model'] == model) & (df['condition'] == cond)][metric].dropna()
        ax.text(pos, ax_ylim[0] + y_span * 0.01,
                f"n={len(vals)}", ha='center', va='bottom', fontsize=6.5, color='#555')

    for mc, model in zip(model_centres, MODEL_ORDER):
        ax.text(mc, ax_ylim[0] - y_span * 0.09,
                model, ha='center', va='top',
                fontsize=10, fontweight='bold', color='#2c2c2c')

    for model in MODEL_ORDER:
        cps = [pos for m, c, pos in positions if m == model]
        x0  = cps[0]  - BOX_W / 2 - 0.06
        x1  = cps[-1] + BOX_W / 2 + 0.06
        y_b = ax_ylim[0] - y_span * 0.05
        ax.annotate('', xy=(x0, y_b), xytext=(x1, y_b),
                    xycoords='data', textcoords='data',
                    arrowprops=dict(arrowstyle='-', color='#cccccc', lw=1.0),
                    annotation_clip=False)

    ax.set_xlim(min(x_positions) - BOX_W, max(x_positions) + BOX_W + 0.3)
    ax.set_xticks([])
    ax.set_ylabel(ylabel, fontsize=9)
    # Raise the panel title (pad 6 -> 24) and drop the italic descriptor to just
    # above the axes so the two no longer overlap.
    ax.set_title(title, fontsize=10, pad=24, fontweight='bold')
    ax.text(0.5, 1.012, subtitle,
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=7.0, color='#666666', style='italic')
    ax.grid(axis='y', ls='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=8.5)

cond_handles = [
    mpatches.Patch(facecolor=COND_COLORS[c], alpha=0.78, label=c)
    for c in COND_ORDER
]
base_h = mlines.Line2D([], [], color='#E24B4A', ls='--', lw=1.8,
                        label='Baseline statute grand mean (avg. of 3 statutes)')
fig.legend(
    handles=cond_handles + [base_h],
    title='Prompt condition',
    title_fontsize=9,
    loc='lower center',
    ncol=4,
    fontsize=9.5,
    framealpha=0.93,
    edgecolor='#cccccc',
    bbox_to_anchor=(0.5, 0.015)
)

fig.suptitle(
    'Stage 3 — Complexity Compression: Structural Degradation Across Prompt Conditions\n'
    'Box = IQR  ·  Whiskers = 1.5×IQR  ·  White bar = median  ·  '
    'n = 60 per condition (30 per model)',
    fontsize=11.5, y=0.965, va='top'
)

output_png = os.path.join(FIG_DIR, 'Fig02_Stage3_Complexity_Compression.png')
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"Saved: {output_png}")
plt.show()

print("Generating Validation Plot for Sentence Length Variance Reduction...")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

plt.figure(figsize=(10, 6))

ax2 = sns.boxplot(
    data=df, 
    x='condition', 
    y='sent_var_red', 
    hue='model',
    order=COND_ORDER,
    palette=['#1f77b4', '#ff7f0e'], # Clean blue and orange for contrast
    linewidth=1.5,
    fliersize=4 # Size of the outlier dots
)

plt.title('The Collapse of Syntactic Variance Across AI Models\n(0% = each domain\'s baseline statute variance)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Prompt Constraint Level', fontsize=14, fontweight='bold')
plt.ylabel('Sentence Length Variance Reduction', fontsize=14, fontweight='bold')

ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0, symbol='%'))

plt.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Domain statute baseline (0% per domain)')

# loc moved from 'lower left' (which sat on top of the boxplots) to the empty upper-right
plt.legend(title='AI Model', title_fontsize='13', fontsize='12', loc='upper right', framealpha=0.95)

plt.tight_layout()

output_var_png = os.path.join(FIG_DIR, 'Fig03_Stage3_Variance_Reduction.png')
plt.savefig(output_var_png, dpi=300, bbox_inches='tight')
print(f"Successfully generated and saved '{output_var_png}'")
plt.show()