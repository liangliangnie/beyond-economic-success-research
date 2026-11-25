# Beyond Economic Success

**Predicting Life Satisfaction Through Economic, Social, and Personal Factors**

A research project using World Values Survey (WVS) Wave 7(https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp) to explore and predict self-reported life satisfaction (Q49) from economic, demographic, social, and personal survey responses. The repository includes the research notebooks, a Streamlit interactive app, the fitted prediction model, figures, and the WVS Wave 7 codebook reference.

> This is a pooled cross-sectional predictive analysis. Model importance and associations should not be interpreted as causal effects or definitive assessments of an individual’s wellbeing.

## Project contents

```text
.
├── app.py                       # Streamlit app (v11; tested questionnaire state fix)
├── assets/                      # Images used by the app
├── models/                      # Fitted reduced-input model and metrics/manifest
├── requirements.txt              # App dependencies
├── reports/                      # Final report pdf
├── notebooks/                    # Analysis workflow, in sequence
├── outputs/
│   ├── app_figures/              # Figures used in the app
│   └── research_figures/         # Available research figures and summary CSV
├── references/                   # WVS Wave 7 codebook/report
└── data/README.md                # Data access and local placement instructions
```

## Interactive app

The app predicts life satisfaction on the WVS 1–10 scale using a reduced set of 19 survey questions plus Country/region (20 model inputs). Country/region must be selected from the 66 categories represented in the model training data. Skipped answers are handled by the model’s fitted imputation pipeline.

The app uses a separately fitted reduced-input Random Forest model, not the Top 30 model discussed in the feature-count experiment. The Top 30 configuration performed slightly better in the reported 5-fold cross-validation (R² 0.4533 vs. 0.4460 for Top 20), while the app uses fewer questions to make the experience shorter.

### Run locally

Python 3.10+ is recommended. From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Deploy with Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the repository.
3. Set the entry point to `app.py`.
4. Deploy.

The app requires the files in `assets/` and `models/` to remain at their repository-relative paths.

## Research workflow

The notebooks are numbered to reflect the project sequence: data quality and variable selection, variable inventory, special-code inspection, feature discovery, EDA, correlation analysis, feature engineering, feature-count/model experiments, legacy model comparison, and the final research narrative.

Some notebooks require the original WVS CSV and/or processed train/test data. These respondent-level data files are **not included** in this public-ready package; see [`data/README.md`](data/README.md) for placement instructions. Notebook outputs may need to be regenerated when running the analysis from scratch.

## Key project figures

The `outputs/` directory includes the app’s final figures and available research charts. The Q50-by-Q288 descriptive chart was regenerated from the supplied processed train/test splits, combining the two splits and retaining valid 1–10 responses. One category-distribution figure referenced in the original narrative notebook was not present among the supplied files, so its image block was omitted rather than replaced with an unrelated chart.

## Important evaluation note

The app model’s supplied test split was used in earlier project analyses, so its test metrics are supplementary rather than a fresh independent holdout. The project’s cross-validation comparisons are the main basis for comparing feature-count configurations.

## Data and citation

The analysis uses World Values Survey Wave 7. Please consult the official WVS portal for data access, documentation, and current citation/redistribution requirements before sharing data or reproducing the analysis.

## License

No software or data license has been added because the project’s ownership and redistribution terms should be confirmed by the author before publication.
