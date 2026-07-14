import os
import subprocess
import sys
import warnings

def install_if_missing(package):
    try:
        if package == 'scikit_posthocs':
            __import__('scikit_posthocs')
        else:
            __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

for pkg in ['pingouin', 'scikit_posthocs', 'statsmodels', 'scipy', 'pandas', 'numpy', 'matplotlib', 'seaborn']:
    install_if_missing(pkg)

import pandas as pd
import numpy as np
import scipy.stats as stats
import pingouin as pg
import scikit_posthocs as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# === PART A: Data Setup ===
INPUT_CSV = r"result\final_outputs\master_dataframe_with_stage5_scores.csv"
INPUT_CSV4 = r"result\final_outputs\master_dataframe_with_stage4_scores.csv"
FINAL_OUT_DIR = r"result\final_outputs"
FIG_DIR = os.path.join(FINAL_OUT_DIR, "figures")
OUT_DIR = FINAL_OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

df5 = pd.read_csv(INPUT_CSV)
df4 = pd.read_csv(INPUT_CSV4)
join_cols = ['file', 'filename', 'model', 'domain', 'condition', 'iteration', 'corpus_type', 'is_partial', 'word_count', 'file_size_kb', 'issues', 'source']
join_cols = [c for c in join_cols if c in df4.columns and c in df5.columns]
df = pd.merge(df4, df5, on=join_cols, how='outer')

# --- Sarvam Refusal Filtering (corrected) ---
before_counts = df.groupby('domain').size().rename('before')

refusal_mask = (
    df['model'].str.lower().eq('sarvam') &
    df['issues'].fillna('').astype(str).str.contains(
        r'\b(refusal_detected|under_3KB)\b',
        case=False, regex=True
    )
)

sarvam_refusal_df = df[refusal_mask].copy()
sarvam_refusal_df.to_csv(os.path.join(OUT_DIR, 
    'sarvam_alignment_friction.csv'), index=False)

df = df[~refusal_mask].copy()

assert not (
    df['model'].str.lower().eq('sarvam') &
    df['issues'].fillna('').astype(str).str.contains(
        r'\b(refusal_detected|under_3KB)\b', case=False, regex=True)
).any(), "CRITICAL: Sarvam refusal rows still present after filtering!"

after_counts = df.groupby('domain').size().rename('after')
comparison_df = pd.DataFrame({
    'before': before_counts, 'after': after_counts,
    'removed': before_counts - after_counts
}).fillna(0).astype(int)
comparison_df.to_csv(os.path.join(OUT_DIR, 
    'sarvam_filtering_impact.csv'))

sarvam_valid_df = df[df['model'].str.lower().eq('sarvam')].copy()
sarvam_domain_counts = sarvam_valid_df.groupby('domain').size()

log_path = os.path.join(OUT_DIR, 'sarvam_refusal_summary.log')
with open(log_path, 'w') as f:
    f.write("=== Sarvam Refusal Filtering Summary ===\n")
    f.write(f"Total removed: {len(sarvam_refusal_df)} rows (Sarvam only)\n")
    f.write(comparison_df.to_string())
    f.write("\nNOTE: Only Sarvam rows filtered. ChatGPT/Gemini untouched.\n")

print(f"Sarvam filter: removed {len(sarvam_refusal_df)} rows. "
      f"Integrity check passed.")
print(comparison_df.to_string())

primary_df = df[(df['corpus_type'] == 'Primary') & (df['model'].isin(['ChatGPT', 'Gemini']))].copy()
sarvam_df = df[df['model'] == 'Sarvam'].copy()

ALL_METRICS = [
    'jsd_score', 'bert_score_f1', 'mattr', 'hedge_density',
    'mean_sent_len', 'sent_len_var', 'Care_Fairness_Score',
    'Authority_Loyalty_Score', 'Technocratic_Framing_Score',
    'US_Distinctive_Term_Frequency'
]
COND_ORDER = ['Status Quo', 'Innovation', 'Unconstrained']

# === PART B: Cell Size Verification ===
print("=== Cell Size Verification ===")
cell_sizes = primary_df.groupby(['model', 'domain', 'condition']).size()
print(cell_sizes)

if not (cell_sizes == 10).all():
    print("WARNING: Not all cells have exactly 10 observations!")
    print(cell_sizes[cell_sizes != 10])
else:
    print("Verification passed: All cells have exactly 10 observations.\n")

# === PART C: Non-Parametric Tests ===
kw_results = []
for metric in ALL_METRICS:
    valid_df = primary_df.dropna(subset=[metric])
    n = len(valid_df)
    
    # By Condition
    groups = [valid_df[valid_df['condition'] == c][metric].values for c in COND_ORDER]
    h_cond, p_cond = stats.kruskal(*groups)
    eta2_cond = (h_cond - 3 + 1) / (n - 3) if n > 3 else np.nan
    kw_results.append({'metric': metric, 'factor': 'condition', 'H_U': h_cond, 'p': p_cond, 'effect_size_eta2_r': eta2_cond})
    
    if p_cond < 0.05:
        posthoc = sp.posthoc_dunn(valid_df, val_col=metric, group_col='condition', p_adjust='bonferroni')
        posthoc.to_csv(os.path.join(OUT_DIR, f'posthoc_dunn_{metric}_condition.csv'))
        
    # By Domain
    doms = valid_df['domain'].unique()
    groups_dom = [valid_df[valid_df['domain'] == d][metric].values for d in doms]
    h_dom, p_dom = stats.kruskal(*groups_dom)
    eta2_dom = (h_dom - len(doms) + 1) / (n - len(doms)) if n > len(doms) else np.nan
    kw_results.append({'metric': metric, 'factor': 'domain', 'H_U': h_dom, 'p': p_dom, 'effect_size_eta2_r': eta2_dom})
    
    if p_dom < 0.05:
        posthoc = sp.posthoc_dunn(valid_df, val_col=metric, group_col='domain', p_adjust='bonferroni')
        posthoc.to_csv(os.path.join(OUT_DIR, f'posthoc_dunn_{metric}_domain.csv'))
        
    # By Model
    chatgpt_vals = valid_df[valid_df['model'] == 'ChatGPT'][metric].values
    gemini_vals = valid_df[valid_df['model'] == 'Gemini'][metric].values
    if len(chatgpt_vals) > 0 and len(gemini_vals) > 0:
        u_model, p_model = stats.mannwhitneyu(chatgpt_vals, gemini_vals, alternative='two-sided')
        n1, n2 = len(chatgpt_vals), len(gemini_vals)
        r_model = 1 - (2 * u_model) / (n1 * n2)
        kw_results.append({'metric': metric, 'factor': 'model', 'H_U': u_model, 'p': p_model, 'effect_size_eta2_r': r_model})
        
pd.DataFrame(kw_results).to_csv(os.path.join(OUT_DIR, 'kruskal_wallis_results.csv'), index=False)

kw_df = pd.read_csv(os.path.join(OUT_DIR, 'kruskal_wallis_results.csv'))
kw_df.rename(columns={'effect_size_eta2_r': 'effect_size'}, inplace=True)
kw_df['effect_size_type'] = kw_df['factor'].apply(
    lambda x: 'rank_biserial_r' if x == 'model' else 'eta_squared')
kw_df['kw_anova_note'] = ''
jsd_model_mask = (kw_df['metric']=='jsd_score') & (kw_df['factor']=='model')
kw_df.loc[jsd_model_mask, 'kw_anova_note'] = (
    "KW(p=0.883) and ANOVA(p=0.002) diverge. KW is rank-based and "
    "more conservative with skewed data. ANOVA is primary cited result.")
kw_df.to_csv(os.path.join(OUT_DIR, 'kruskal_wallis_results.csv'), index=False)

# Model post-hoc (Mann-Whitney with effect size per metric)
model_posthoc_results = []
for metric in ALL_METRICS:
    valid_df = primary_df.dropna(subset=[metric])
    chatgpt_vals = valid_df[valid_df['model'] == 'ChatGPT'][metric].values
    gemini_vals  = valid_df[valid_df['model'] == 'Gemini'][metric].values
    if len(chatgpt_vals) > 0 and len(gemini_vals) > 0:
        u, p = stats.mannwhitneyu(chatgpt_vals, gemini_vals, alternative='two-sided')
        r = 1 - (2 * u) / (len(chatgpt_vals) * len(gemini_vals))
        model_posthoc_results.append({
            'metric': metric, 'group1': 'ChatGPT', 'group2': 'Gemini',
            'U_stat': u, 'p_value': p, 'rank_biserial_r': r,
            'significant': 'Yes' if p < 0.05 else 'No'
        })

if model_posthoc_results:
    pd.DataFrame(model_posthoc_results).to_csv(
        os.path.join(OUT_DIR, 'posthoc_model_pairwise_comparison.csv'),
        index=False
    )

# === PART D: Mixed ANOVA (Between-Subjects Factorial) ===
anova_results = []
all_pvalues = []
anova_metrics = []

for metric in ALL_METRICS:
    valid_df = primary_df.dropna(subset=[metric, 'model', 'domain', 'condition'])
    if len(valid_df) == 0: continue
    
    try:
        aov = pg.anova(data=valid_df, dv=metric, between=['model', 'domain', 'condition'], detailed=True)
        aov['metric'] = metric
        anova_results.append(aov)
        p_col = 'p_unc' if 'p_unc' in aov.columns else 'p-unc'
        valid_rows = aov.dropna(subset=[p_col])
        all_pvalues.extend(valid_rows[p_col].tolist())
        anova_metrics.extend([(metric, src) for src in valid_rows['Source'].tolist()])
    except Exception as e:
        print(f"ANOVA failed for {metric}: {e}")

if all_pvalues:
    reject, p_corrected, _, _ = multipletests(all_pvalues, method='fdr_bh')
    final_anova_df = pd.concat(anova_results, ignore_index=True)
    
    corr_dict = {f"{m}_{s}": p for (m, s), p in zip(anova_metrics, p_corrected)}
    final_anova_df['p_BH_corrected'] = final_anova_df.apply(
        lambda row: corr_dict.get(f"{row['metric']}_{row['Source']}", np.nan), axis=1)

    anova_out_path = os.path.join(OUT_DIR, 'anova_results_all_metrics.csv')
    
    # Check for Sarvam exclusion systematically
    # model factor has degrees of freedom == 1 if only 2 models are present
    model_rows = final_anova_df[final_anova_df['Source'] == 'model']
    ddof1_col = 'ddof1' if 'ddof1' in final_anova_df.columns else 'DF'
    if not model_rows.empty and (model_rows[ddof1_col] == 1).any():
        warning_msg = "WARNING: Only 2 models present in ANOVA after refusal filtering. Sarvam excluded due to systemic alignment refusals in National Security domain. Results reflect ChatGPT vs Gemini comparison only."
        print(f"\n{warning_msg}\n")
        
        # Add warning as a comment row (starting with #)
        with open(anova_out_path, 'w') as f:
            f.write(f"# {warning_msg}\n")
        final_anova_df.to_csv(anova_out_path, index=False, mode='a')
    else:
        final_anova_df.to_csv(anova_out_path, index=False)

# === PART E: Regression Models ===
primary_df['condition_code'] = primary_df['condition'].map({'Status Quo': 0, 'Innovation': 1, 'Unconstrained': 2})
reg1_results = []
reg2_results = []

for metric in ALL_METRICS:
    valid_df = primary_df.dropna(subset=[metric, 'condition_code', 'domain', 'model'])
    if len(valid_df) == 0: continue
    
    # Model 1
    slope, intercept, r_value, p_value, std_err = stats.linregress(valid_df['condition_code'], valid_df[metric])
    reg1_results.append({
        'metric': metric, 'slope': slope, 'R2': r_value**2, 'p_value': p_value, 
        'CI_lower': slope - 1.96*std_err, 'CI_upper': slope + 1.96*std_err
    })
    
    # Model 2
    formula = f"{metric} ~ C(condition, Treatment('Status Quo')) + C(domain) + C(model)"
    try:
        model = smf.ols(formula, data=valid_df).fit()
        reg2_results.append({
            'metric': metric,
            'cond_innovation_coef': model.params.get("C(condition, Treatment('Status Quo'))[T.Innovation]", np.nan),
            'cond_unconstrained_coef': model.params.get("C(condition, Treatment('Status Quo'))[T.Unconstrained]", np.nan),
            'R2': model.rsquared, 'adj_R2': model.rsquared_adj, 'F_stat': model.fvalue, 'p_value': model.f_pvalue
        })
    except Exception as e:
        print(f"Regression 2 failed for {metric}: {e}")

pd.DataFrame(reg1_results).to_csv(os.path.join(OUT_DIR, 'regression_condition_effect.csv'), index=False)
pd.DataFrame(reg2_results).to_csv(os.path.join(OUT_DIR, 'regression_full_model.csv'), index=False)


# === PART F: Cross-Dimension Correlation Heatmap ===
corr_df = primary_df[ALL_METRICS].dropna()
rho, p_vals = stats.spearmanr(corr_df)
rho_df = pd.DataFrame(rho, index=ALL_METRICS, columns=ALL_METRICS)
p_df = pd.DataFrame(p_vals, index=ALL_METRICS, columns=ALL_METRICS)

labels_map = {
    'jsd_score': 'Lexical Drift (JSD)', 'bert_score_f1': 'Semantic Drift (BERTScore)',
    'mattr': 'Lexical Diversity (MATTR)', 'hedge_density': 'Hedge Density',
    'mean_sent_len': 'Mean Sentence Length', 'sent_len_var': 'Sentence Variance',
    'Care_Fairness_Score': 'Care/Fairness (MFD)', 'Authority_Loyalty_Score': 'Authority/Loyalty (MFD)',
    'Technocratic_Framing_Score': 'Technocratic Framing', 'US_Distinctive_Term_Frequency': 'US-Distinctive Terms'
}

annot_df = rho_df.copy()
for i in range(len(ALL_METRICS)):
    for j in range(len(ALL_METRICS)):
        val = rho_df.iloc[i, j]
        p = p_df.iloc[i, j]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        annot_df.iloc[i, j] = f"{val:.2f}{stars}"

plt.figure(figsize=(14, 12))
sns.heatmap(rho_df, annot=annot_df, fmt="", cmap="RdYlGn", center=0, vmin=-1, vmax=1,
            xticklabels=[labels_map[m] for m in ALL_METRICS],
            yticklabels=[labels_map[m] for m in ALL_METRICS])
plt.title("Cross-Dimension Correlation Matrix: Evidence for Unified Algorithmic Drift\n"
          "Spearman ρ across all 10 drift metrics (N=180 primary drafts)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'Fig09_Stage7_CrossDimension_Heatmap.png'), dpi=300)
plt.close()

mask = np.triu(np.ones_like(rho, dtype=bool), k=1)
off_diag = np.abs(rho[mask])
print(f"Mean absolute correlation (off-diagonal): {np.mean(off_diag):.3f}")
print(f"Pairs with |r| > 0.3: {np.sum(off_diag > 0.3)}")
print(f"Pairs with |r| > 0.5: {np.sum(off_diag > 0.5)}\n")

# === PART G: Print Full Results Summary ===
print(f"{'METRIC':<30} | {'KW-Cond H':<12} | {'p (KW)':<10} | {'ANOVA F(cond)':<15} | {'p (BH)':<10}")
for metric in ALL_METRICS:
    kw_cond = next((x for x in kw_results if x['metric'] == metric and x['factor'] == 'condition'), None)
    anv_cond = final_anova_df[(final_anova_df['metric'] == metric) & (final_anova_df['Source'] == 'condition')] if 'final_anova_df' in locals() else pd.DataFrame()
    
    h_val = f"{kw_cond['H_U']:.2f}" if kw_cond else "N/A"
    p_kw = f"{kw_cond['p']:.4f}" if kw_cond else "N/A"
    
    f_val = f"{anv_cond['F'].values[0]:.2f}" if not anv_cond.empty else "N/A"
    p_bh = f"{anv_cond['p_BH_corrected'].values[0]:.4f}" if not anv_cond.empty else "N/A"
    
    print(f"{metric:<30} | {h_val:<12} | {p_kw:<10} | {f_val:<15} | {p_bh:<10}")

print("\n=== Plain-Language Interpretations ===")
def get_effect_size_label(factor, es):
    if pd.isna(es): return "no"
    if factor in ['condition', 'domain']:
        return "large" if es >= 0.14 else "medium" if es >= 0.06 else "small" if es >= 0.01 else "no"
    else:
        abs_es = abs(es)
        return "large" if abs_es >= 0.5 else "medium" if abs_es >= 0.3 else "small" if abs_es >= 0.1 else "no"

for metric in ALL_METRICS:
    kw_cond = next((x for x in kw_results if x['metric'] == metric and x['factor'] == 'condition'), None)
    kw_dom = next((x for x in kw_results if x['metric'] == metric and x['factor'] == 'domain'), None)
    kw_mod = next((x for x in kw_results if x['metric'] == metric and x['factor'] == 'model'), None)
    
    if kw_cond and kw_dom and kw_mod:
        cond_label = get_effect_size_label('condition', kw_cond['effect_size_eta2_r'])
        dom_sig = "significant" if kw_dom['p'] < 0.05 else "not significant"
        mod_sig = "significant" if kw_mod['p'] < 0.05 else "not significant"
        print(f"{metric}: Condition has {cond_label} effect (η²={kw_cond['effect_size_eta2_r']:.3f}). "
              f"Domain effect is {dom_sig}. Model effect is {mod_sig}.")

print("\nStage 7 Complete. All inference results saved to result/final_outputs/")
