# Human and AI in Policy Drafting

A quantitative study of **how AI-generated legislation diverges from human-drafted
statutes** — and by how much. We prompt large language models to draft policy in
three legal domains, then measure the "drift" of their output away from the real
enacted baseline statutes across ten linguistic, structural, and ideological
metrics, and test which differences are statistically significant.

## Research question

When an LLM is asked to draft policy, does its output differ from human legislative
drafting — semantically, structurally, and ideologically — and can that difference
be quantified? A secondary question compares a **sovereign Indian model (Sarvam)**
against Western frontier models to look for jurisdictional and refusal-behaviour
differences.

## Experimental design

| Factor | Levels |
|---|---|
| **Model** | ChatGPT, Gemini (primary) · Sarvam (supplementary / sovereign) |
| **Legal domain** | Air Pollution, Data Protection, National Security |
| **Prompt condition** | Status Quo, Innovation, Unconstrained |
| **Iterations** | 10 per cell |

This yields **180 primary drafts** (2 models × 3 domains × 3 conditions × 10) plus
a Sarvam supplementary set. Each AI draft is compared against the corresponding
enacted baseline statute (e.g. the Air Act 1981, the DPDP Act 2023, the National
Security Act 1980) and a broader corpus of related legislation.

## Drift dimensions measured

- **Semantic drift** — BERTScore F1 against the baseline statute
- **Lexical drift** — Jensen–Shannon divergence, TF-IDF distance
- **Lexical diversity** — MATTR (moving-average type–token ratio)
- **Structural compression** — mean sentence length, sentence-length variance, hedge density
- **Ideological framing** — Moral Foundations (Care/Fairness vs Authority/Loyalty), technocratic framing, log-odds lexical framing
- **Jurisdictional drift** — US-distinctive term frequency, GDPR similarity
- **Refusal / completeness behaviour** — short-output and refusal rates (esp. Sarvam, National Security)

Group differences are tested with Kruskal–Wallis, one-way ANOVA, Dunn's post-hoc
(Bonferroni), and OLS regression. See `result/final_outputs/01_statistical_results.csv`
and `result/statistical_inference/`.

## Repository layout

```
.
├── result/                         # ← all analysis outputs (figures + CSVs)
│   ├── final_outputs/
│   │   ├── figures/                # Fig01–Fig11 (the paper figures)
│   │   ├── 01_statistical_results.csv
│   │   ├── 02_posthoc_dunn_results.csv
│   │   ├── 03_sarvam_analysis.csv
│   │   └── 04_master_data.csv      # consolidated per-draft metric table
│   ├── statistical_inference/      # ANOVA / Kruskal / Dunn / regression tables
│   ├── sarvam_analysis/            # sovereign-model report + tables
│   ├── jurisdictional_heatmaps/
│   ├── archive/                    # intermediate stage dataframes
│   └── master_dataframe_with_stage{3,4,5}_scores.csv
│
├── pdftotxt.py, segPDF.py, docxtotxt.py   # text extraction from source PDFs/DOCX
├── datacollectionjurisdictional.py        # build the jurisdictional corpus
├── datacleaning.py                        # assemble the master dataframe
├── semantic.py                 # Stage 2 — semantic & lexical drift  → Fig01
├── complexity_compression.py   # Stage 3 — structural compression     → Fig02, Fig03
├── ideologicalbias.py          # Stage 4 — moral foundations / framing → Fig04–Fig08
├── jurisdictionaldrift.py      # jurisdictional drift (US terms, GDPR) → heatmaps
├── hallvsinnovation.py         # hallucination-vs-innovation topic modelling
├── statistical_inference.py    # Stage 7 — inferential statistics      → Fig09
├── sarvam_analysis.py, sarvamnatdata.py   # Stage 8 — sovereign model  → Fig10, Fig11
├── regenerate_figures.py       # re-render Fig01–Fig03 from saved CSVs (see below)
├── fix_filename_typos.py, fix_results_audit.py, cleanup_comments.py   # maintenance utilities
├── exclusion_register.json     # drafts excluded (refusals / truncated outputs)
├── requirements.txt
└── .env.example
```

## The figures

| File | What it shows |
|---|---|
| `Fig01_Stage2_Semantic_Lexical_Drift.png` | Semantic similarity vs lexical drift (r = −0.92) |
| `Fig02_Stage3_Complexity_Compression.png` | MATTR, hedge density, sentence length & variance vs statute |
| `Fig03_Stage3_Variance_Reduction.png` | Collapse of sentence-length variance |
| `Fig04_Stage4_LogOdds_Framing.png` | Log-odds lexical framing, AI vs baseline |
| `Fig05_Stage4_MFD_Radar.png` | Moral-foundations profile by condition |
| `Fig06_Stage4_MFD_NationalSecurity.png` | Care/Fairness vs Authority/Loyalty in national-security drafts |
| `Fig07/08_Stage4_*_Interaction.png` | Condition × domain interaction plots |
| `Fig09_Stage7_CrossDimension_Heatmap.png` | Cross-dimension correlation matrix |
| `Fig10/11_Stage8_Sarvam_*.png` | Sovereign vs Western model comparison |

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate     |  macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm        # needed for sentence-level metrics
cp .env.example .env                           # then add your SARVAM_API_KEY
```

### Regenerating the figures

The three Stage 2/3 figures can be re-rendered directly from the saved master
data (no heavy NLP recompute needed):

```bash
python regenerate_figures.py
```

The full analysis scripts (`semantic.py`, `complexity_compression.py`, …) reproduce
the metrics end-to-end but require the raw text corpus (see below) plus the full
dependency stack.

## Data availability

To keep the repository lightweight, the **raw source corpus is not included**: the
original legislation PDFs, the extracted plain text (`extracted_text/`), and the
generated AI drafts (`chatgpt_policy_drafts/`, `gemini outputs/`, `output/`). These
are excluded via `.gitignore`. Every computed metric that the analysis depends on is
preserved in `result/` (notably `result/final_outputs/04_master_data.csv`), so the
statistics and figures are fully reproducible from what is committed here. The source
documents are public legislation and can be re-collected; contact the author for the
exact corpus snapshot.

## Notes

- `.env` holds the Sarvam API key and is gitignored — never commit it.
- Some Sarvam National-Security drafts were refusals or truncated outputs; these are
  logged in `exclusion_register.json` and excluded from the relevant analyses.
