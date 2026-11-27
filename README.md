# Beyond Economic Success

*A Predictive Analysis of Life Satisfaction Using World Values Survey Wave 7*

## Project overview

Life satisfaction is shaped by more than economic success. This project uses survey data and machine learning to explore how economic, social, demographic, and personal factors relate to self-reported life satisfaction across countries and regions.

The project asks:

> To what extent can economic, demographic, social, and personal factors explain and predict differences in self-reported life satisfaction?

The analysis is predictive and exploratory. It identifies patterns and associations in cross-sectional survey data; it does not establish causal relationships.

## Data

The project uses **World Values Survey (WVS), Wave 7**.

- Approximately 97,220 respondent records
- 613 original survey variables
- 66 country and region entries
- Target: **Q49 — Life Satisfaction**, measured on a 1–10 scale

The dataset contains special missing-value codes. These are handled during data preparation rather than treated as valid responses.

## Analysis workflow

The research workflow includes:

1. Data inventory and assessment of survey variables
2. Handling of special codes and missing responses
3. Exploratory data analysis and descriptive comparisons
4. Feature assessment and feature selection
5. Feature engineering and model experiments
6. Model evaluation and interpretation
7. Development of an interactive Streamlit application

Several model configurations were explored in the research notebooks, including feature-count experiments and Random Forest models. Feature interpretation uses multiple approaches, including Random Forest feature importance, permutation importance, and SHAP.

## Interactive app

The Streamlit app allows visitors to explore the project and make a model-based prediction of life satisfaction using a subset of survey questions.

The prediction model uses **19 survey inputs** and does not require a country selection. This makes the survey simpler to complete and avoids limiting the user to a country list. Unanswered inputs are imputed using global median values.

The app includes:

- A project overview
- An interactive prediction survey
- Model insights and feature interpretation
- Additional descriptive insights about age, gender, and country/region patterns
- A research workflow overview

The app's prediction is an estimate from a statistical model, not a definitive assessment of an individual's wellbeing.

## Selected model performance

On the app model's evaluation test split:

| Metric | Value |
|---|---:|
| R² | 0.429 |
| MAE | 1.235 |
| RMSE | 1.686 |
| Test observations | 19,340 |

These results should be interpreted as indicative predictive performance. The test split was reused during earlier model comparisons, so it is not a fully untouched independent evaluation set.

The research notebooks also report separate feature-selection and model-comparison experiments. Those results refer to different model configurations and should not be treated as direct evaluations of the interactive app model.

## Key reflections

The project explores questions that go beyond model performance:

- Is feeling financially secure as important as having a higher income?
- Do people evaluate their financial wellbeing through their own circumstances or by comparing themselves with others?
- Can economic success fully capture how people experience wellbeing?

The descriptive analyses also examine how reported life satisfaction and related wellbeing measures vary across age groups, gender groups, and country/region contexts. These are descriptive patterns, not causal conclusions.

## Run locally

Clone or download the repository, then install the dependencies and launch the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

- WVS Wave 7 is cross-sectional; the analysis does not establish causality or describe within-person changes over time.
- Survey responses are self-reported and may reflect differences in interpretation or cultural context.
- The prediction model is trained on pooled respondents and is not designed to establish performance on entirely unseen countries.
- Test-set reuse during earlier model comparisons limits how independently the final test performance can be interpreted.
- Model explanations describe how the model uses features; they do not demonstrate that a feature causes life satisfaction to change.

## Data source

World Values Survey Association — World Values Survey, Wave 7.

Please refer to the WVS documentation and data-use terms when using or redistributing the source data.
