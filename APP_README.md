# Streamlit app notes

The main app is `app.py`. Install the packages listed in `requirements.txt` and run `streamlit run app.py`.

The Prediction page contains a short 19-question survey plus Country/region. It is not the complete WVS questionnaire. The model bundle is stored in `models/app_top20_model.joblib`; the `assets/` directory contains all app visuals.

The v11 update fixed the saved Country/region state across survey sections. The model and visual assets were preserved from the v11 app package.
