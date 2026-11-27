# Beyond Economic Success — v18

Streamlit research app using World Values Survey Wave 7.

## v18 updates
- Removed Country/region from the prediction survey and retrained the app Random Forest using 19 survey questions only.
- Skipped answers are filled with global median values from valid WVS responses; Country is not used as a predictor or imputation group.
- Recreated age, gender-by-age, country life-satisfaction, and country gender-gap charts as interactive Plotly charts using weighted descriptive summaries.
- Kept the existing Warm Sun Modern visual design and page structure.
- Replaced several static figure presentations with responsive HTML figure cards. Original SHAP/ranking source figures remain embedded because their original plotting arrays were not retained in the app bundle.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model
The app model predicts WVS Q49 (life satisfaction, 1–10) from 19 survey inputs. Country is not an input. This is a pooled cross-sectional predictive model, not a causal or individual clinical assessment.
