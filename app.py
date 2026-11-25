from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

APP_DIR = Path(__file__).parent
ASSET_DIR = APP_DIR / "assets"

# v13 hotfix: restore shared section_head and reflection_card helpers
st.set_page_config(
    page_title="Beyond Economic Success",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TOP20_FEATURES = [
    "Q50", "Country", "Q48", "Q47", "Q251", "Q252", "Q288", "Q163",
    "Q106", "Q158", "Q159", "Q249", "Q250", "Q164", "Q287", "Q131",
    "Q80", "Q253", "Q75", "Q88",
]
TOP20_SURVEY_FEATURES = [f for f in TOP20_FEATURES if f != "Country"]
QUESTION_GROUPS = [
    ("01", "Wellbeing & context", TOP20_SURVEY_FEATURES[:7],
     "Economic wellbeing, health and perceptions of public institutions."),
    ("02", "Society & outlook", TOP20_SURVEY_FEATURES[7:13],
     "Views on income, science, democracy and personal beliefs."),
    ("03", "Values & community", TOP20_SURVEY_FEATURES[13:],
     "Social class, neighborhood safety and confidence in organizations."),
]

APP_MODEL_PATH = APP_DIR / "models" / "app_top20_model.joblib"

QUESTION_INFO = {
    "Q50": ("Financial satisfaction", "How satisfied are you with the financial situation of your household?", "1 = completely dissatisfied · 10 = completely satisfied", list(range(1, 11)), {1:"Completely dissatisfied",10:"Completely satisfied"}),
    "Q48": ("Freedom of choice and control", "How much freedom of choice and control do you feel you have over the way your life turns out?", "1 = none at all · 10 = a great deal", list(range(1, 11)), {1:"None at all",10:"A great deal"}),
    "Q47": ("Self-rated health", "All in all, how would you describe your state of health these days?", "1 = very good · 5 = very poor", [1,2,3,4,5], {1:"Very good",2:"Good",3:"Fair",4:"Poor",5:"Very poor"}),
    "Q251": ("How democratically the country is governed", "How democratically is this country being governed today?", "1 = not at all democratic · 10 = completely democratic", list(range(1, 11)), {1:"Not at all democratic",10:"Completely democratic"}),
    "Q252": ("Satisfaction with political system", "How satisfied are you with how the political system is functioning in your country these days?", "1 = not satisfied at all · 10 = completely satisfied", list(range(1, 11)), {1:"Not satisfied at all",10:"Completely satisfied"}),
    "Q288": ("Household income position", "Where does your household fall on the income scale within your country?", "1 = lowest income group · 10 = highest income group", list(range(1, 11)), {1:"Lowest income group",10:"Highest income group"}),
    "Q163": ("Science and technology: better or worse for the world", "All things considered, is the world better off or worse off because of science and technology?", "1 = a lot worse off · 10 = a lot better off", list(range(1, 11)), {1:"A lot worse off",10:"A lot better off"}),
    "Q106": ("Income equality vs. larger income differences", "Which view is closer to yours?", "1 = incomes more equal · 10 = larger income differences", list(range(1, 11)), {1:"Incomes more equal",10:"Larger income differences"}),
    "Q158": ("Science and technology improve everyday life", "Science and technology are making our lives healthier, easier, and more comfortable.", "1 = completely disagree · 10 = completely agree", list(range(1, 11)), {1:"Completely disagree",10:"Completely agree"}),
    "Q159": ("Science and technology create future opportunities", "Because of science and technology, there will be more opportunities for the next generation.", "1 = completely disagree · 10 = completely agree", list(range(1, 11)), {1:"Completely disagree",10:"Completely agree"}),
    "Q249": ("Women’s equal rights as a democratic principle", "How essential is it for democracy that women have the same rights as men?", "0 = against democracy · 1 = not essential · 10 = essential", list(range(0, 11)), {0:"Against democracy",1:"Not an essential characteristic",10:"An essential characteristic"}),
    "Q250": ("Importance of living in a democratic country", "How important is it for you to live in a country that is governed democratically?", "1 = not at all important · 10 = absolutely important", list(range(1, 11)), {1:"Not at all important",10:"Absolutely important"}),
    "Q164": ("Importance of God", "How important is God in your life?", "1 = not at all important · 10 = very important", list(range(1, 11)), {1:"Not at all important",10:"Very important"}),
    "Q287": ("Subjective social class", "People sometimes describe themselves as belonging to a class. Which would you say you belong to?", "1 = upper class · 5 = lower class", [1,2,3,4,5], {1:"Upper class",2:"Upper middle class",3:"Lower middle class",4:"Working class",5:"Lower class"}),
    "Q131": ("Security in your neighborhood", "How secure do you feel these days in your neighborhood?", "1 = very secure · 4 = not at all secure", [1,2,3,4], {1:"Very secure",2:"Quite secure",3:"Not very secure",4:"Not at all secure"}),
    "Q80": ("Confidence in women’s organizations", "How much confidence do you have in women’s organizations?", "1 = a great deal · 4 = none at all", [1,2,3,4], {1:"A great deal",2:"Quite a lot",3:"Not very much",4:"None at all"}),
    "Q253": ("Respect for individual human rights", "How much respect is there for individual human rights nowadays in this country?", "1 = a great deal of respect · 4 = no respect at all", [1,2,3,4], {1:"A great deal of respect",2:"Fairly much respect",3:"Not much respect",4:"No respect at all"}),
    "Q75": ("Confidence in universities", "How much confidence do you have in universities?", "1 = a great deal · 4 = none at all", [1,2,3,4], {1:"A great deal",2:"Quite a lot",3:"Not very much",4:"None at all"}),
    "Q88": ("Confidence in the World Health Organization", "How much confidence do you have in the World Health Organization (WHO)?", "1 = a great deal · 4 = none at all", [1,2,3,4], {1:"A great deal",2:"Quite a lot",3:"Not very much",4:"None at all"}),
}

@st.cache_resource
def load_app_model():
    return joblib.load(APP_MODEL_PATH)


def apply_user_imputation(row, bundle):
    features = bundle["features"]
    stats = bundle["imputer_stats"]
    result = {feature: row.get(feature) for feature in features}
    country = row["Country"]
    income = result.get("Q288")
    if income is None or pd.isna(income):
        income = stats["q288_country"].get(country, stats["q288_global"])
        result["Q288"] = income
    for feature in features:
        if feature == "Q288":
            continue
        value = result.get(feature)
        if value is None or pd.isna(value):
            group_value = stats["country_income"][feature].get((country, float(income)), np.nan)
            country_value = stats["country"][feature].get(country, np.nan)
            value = group_value
            if pd.isna(value): value = country_value
            if pd.isna(value): value = stats["global"][feature]
            result[feature] = value
    return result


def country_label(code):
    special = {"HKG":"Hong Kong SAR", "MAC":"Macao SAR", "NIR":"Northern Ireland", "TWN":"Taiwan", "KOS":"Kosovo"}
    if code in special:
        return f"{special[code]} ({code})"
    try:
        import pycountry
        country = pycountry.countries.get(alpha_3=code)
        if country:
            return f"{country.name} ({code})"
    except Exception:
        pass
    return code

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
      --bg:#FBF8F2; --card:#FFFEFC; --ink:#2D211D; --muted:#786B63;
      --orange:#D95B18; --orange-dark:#B94710; --sage:#7D927B;
      --line:#E8DCD0; --pale:#F7E7D9;
    }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:var(--bg); color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    .block-container { max-width:1240px; padding-top:1.2rem; padding-bottom:4rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif!important; color:var(--ink)!important;
      font-weight:800!important; letter-spacing:-.045em!important; }
    h1 { font-size:clamp(2.5rem,5.7vw,4.8rem)!important; line-height:1.04!important; }
    h2 { font-size:clamp(1.8rem,3.2vw,2.6rem)!important; line-height:1.12!important; }
    h3 { font-size:1.25rem!important; }
    p,li,label { color:var(--ink); }
    .st-key-home_header { position:relative; overflow:hidden; background:linear-gradient(110deg,#fff 0%,#fff 62%,#FFF5EB 100%);
      border:1px solid #F0E6DC; border-radius:32px; padding:clamp(1.5rem,4vw,3.2rem);
      margin-bottom:1.7rem; box-shadow:0 2px 0 rgba(66,38,20,.03); }
    .st-key-home_header:before { content:''; pointer-events:none; position:absolute; width:420px;height:420px;right:-90px;top:-125px;
      border-radius:50%; background:radial-gradient(circle,rgba(247,179,111,.22),rgba(247,179,111,0) 70%); }
    .st-key-prediction_header, .st-key-insights_header, .st-key-workflow_header { margin-bottom:1.4rem; }
    .st-key-home_header div[data-testid="stRadio"],
    .st-key-prediction_header div[data-testid="stRadio"],
    .st-key-insights_header div[data-testid="stRadio"],
    .st-key-workflow_header div[data-testid="stRadio"] { position:relative; z-index:2; margin-top:1.15rem; }
    .st-key-home_header div[role="radiogroup"],
    .st-key-prediction_header div[role="radiogroup"],
    .st-key-insights_header div[role="radiogroup"],
    .st-key-workflow_header div[role="radiogroup"] { flex-wrap:wrap; }
    .topline { display:flex; justify-content:space-between; align-items:center; gap:1rem;
      margin-bottom:1.25rem; position:relative; z-index:2; }
    .brand { font-size:.82rem; font-weight:800; letter-spacing:.04em; }
    .brand span,.accent { color:var(--orange); }
    .nav-caption { color:#8A7A70; font-size:.72rem; font-weight:700; letter-spacing:.08em; }
    .hero { position:relative; overflow:visible; background:transparent; border:0; border-radius:0;
      padding:0; min-height:unset; box-shadow:none; }
    .hero:before,.hero:after { display:none; }
    .hero-copy { position:relative; z-index:2; max-width:720px; }
    .eyebrow { display:inline-flex; align-items:center; gap:.5rem; border-radius:999px;
      padding:.5rem .85rem; background:#F7EFE5; color:var(--orange); font-size:.7rem;
      font-weight:800; letter-spacing:.08em; text-transform:uppercase; margin-bottom:1.2rem; }
    .eyebrow:before { content:''; width:7px;height:7px;border-radius:50%;background:var(--orange); }
    .hero-deck { max-width:680px; color:var(--muted); font-size:1.02rem; line-height:1.75; margin-top:1.05rem; margin-bottom:.65rem; }
    .hero-deck + .hero-deck { margin-top:.7rem; }
    .section-head { margin:2.4rem 0 1rem; }
    .section-label { color:var(--orange); text-transform:uppercase; letter-spacing:.12em;
      font-weight:800; font-size:.7rem; margin-bottom:.45rem; }
    .metric-card { background:var(--card); border:1px solid var(--line); border-radius:22px;
      padding:1.3rem 1.45rem; height:100%; box-shadow:0 2px 0 rgba(66,38,20,.025); }
    .metric-value { color:var(--orange); font-family:'Manrope',sans-serif; font-size:1.8rem;
      font-weight:800; letter-spacing:-.05em; margin-bottom:.35rem; }
    .metric-title { font-size:.85rem; font-weight:700; margin-bottom:.25rem; }
    .metric-desc { color:var(--muted); font-size:.78rem; line-height:1.5; }
    .content-card { background:var(--card); border:1px solid var(--line); border-radius:24px;
      padding:1.45rem; height:100%; }
    .soft-note { background:#F0F4EC; border:1px solid #E1E9DB; border-radius:18px;
      padding:1.05rem 1.2rem; color:#53624D; line-height:1.65; }
    .warm-note { background:#FFF1E6; border:1px solid #F4DCCB; border-radius:18px;
      padding:1.05rem 1.2rem; color:#79513D; line-height:1.65; }
    .reflection-card { background:#FFFEFC; border:1px solid var(--line); border-radius:20px;
      padding:1.25rem 1.3rem; height:100%; }
    .reflection-num { color:var(--orange); font-size:.7rem; font-weight:800; letter-spacing:.1em; }
    .reflection-title { font-family:'Manrope',sans-serif; font-size:1.08rem; font-weight:800;
      margin:.45rem 0 .55rem; }
    .reflection-question { color:var(--muted); font-size:.88rem; line-height:1.65; }
    .feature-card { border:1px solid var(--line); background:#FFFEFC; border-radius:17px;
      padding:.95rem 1rem; margin-bottom:.7rem; }
    .feature-code { color:var(--orange); font-size:.76rem; font-weight:800; letter-spacing:.08em; }
    .feature-text { color:var(--muted); font-size:.82rem; line-height:1.5; margin-top:.3rem; }
    .method-item { display:flex; gap:.85rem; align-items:flex-start; padding:.8rem 0;
      border-bottom:1px solid #F0E7DE; }
    .method-item:last-child { border-bottom:0; }
    .method-icon { color:var(--orange); font-size:1.15rem; width:1.5rem; flex:0 0 1.5rem; }
    .method-name { font-weight:800; font-size:.93rem; }
    .method-detail { color:var(--muted); font-size:.82rem; line-height:1.5; margin-top:.18rem; }
    .notebook-wrap { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:.8rem; }
    .notebook-chip { border:1px solid var(--line); background:#FFFEFC; border-radius:999px;
      padding:.55rem .8rem; font-size:.76rem; font-weight:700; color:#5C4D44; }
    .footer { border-top:1px solid var(--line); margin-top:3rem; padding-top:1.2rem;
      color:#98877D; font-size:.7rem; letter-spacing:.08em; }
    div.stButton > button { border-radius:999px; border:1px solid var(--orange);
      background:var(--orange); color:#fff; padding:.68rem 1.25rem; font-weight:700;
      box-shadow:0 7px 15px rgba(218,87,13,.16); }
    div.stButton > button:hover { background:var(--orange-dark); border-color:var(--orange-dark); color:#fff; }
    div.stButton > button[kind="secondary"] { background:#fff; color:var(--ink);
      border-color:var(--line); box-shadow:none; }
    div.stButton > button[kind="secondary"]:hover { background:#FFF4EA; border-color:#F1C8AD; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:18px; padding:1rem; }
    [data-testid="stMetricLabel"] { color:var(--muted); }
    [data-testid="stMetricValue"] { color:var(--orange-dark); font-family:'Manrope',sans-serif; font-weight:800; }
    [data-testid="stProgressBar"] > div > div { background:var(--orange); }
    div[role="radiogroup"] { gap:.45rem; }
    div[role="radiogroup"] label { background:#fff; border:1px solid var(--line); border-radius:999px; padding:.4rem .85rem; }
    div[role="radiogroup"] label:has(input:checked) { background:var(--orange); border-color:var(--orange); color:#fff; }
    div[role="radiogroup"] label:has(input:checked) * { color:#fff!important; }
    hr { border-color:var(--line); }
    @media(max-width:800px) { .st-key-home_header { padding:1.5rem 1.2rem; border-radius:24px; }
      .topline { align-items:flex-start; flex-direction:column; gap:.35rem; }
      .hero-deck {font-size:.96rem;} .block-container {padding-top:.65rem;} }
    </style>
    """,
    unsafe_allow_html=True,
)



def section_head(label, title=None, subtitle=None):
    title_html = f"<h2>{title}</h2>" if title else ""
    subtitle_html = (
        f"<p style='color:#786B63;line-height:1.7;margin-top:.5rem'>{subtitle}</p>"
        if subtitle else ""
    )
    st.markdown(
        f"<div class='section-head'><div class='section-label'>{label}</div>"
        f"{title_html}{subtitle_html}</div>",
        unsafe_allow_html=True,
    )


def reflection_card(num, title, question):
    st.markdown(
        f"<div class='reflection-card'><div class='reflection-num'>{num}</div>"
        f"<div class='reflection-title'>{title}</div>"
        f"<div class='reflection-question'>{question}</div></div>",
        unsafe_allow_html=True,
    )


PAGES = ["Home", "Prediction", "Model Insights", "Workflow"]
if "main_nav" not in st.session_state:
    st.session_state.main_nav = "Home"


def brand_line():
    st.markdown(
        "<div class='topline'><div class='brand'>BEYOND <span>ECONOMIC SUCCESS</span></div></div>",
        unsafe_allow_html=True,
    )


def page_navigation():
    st.radio(
        "Navigate",
        PAGES,
        horizontal=True,
        label_visibility="collapsed",
        key="main_nav",
    )


# The navigation is rendered inside each page's own header. The inner pages
# retain their distinct title, eyebrow and introduction instead of inheriting
# the Home hero.
page = st.session_state.main_nav

if page == "Home":
    with st.container(key="home_header"):
        brand_line()
        st.markdown(
            """
            <div class="hero"><div class="hero-copy">
              <div class="eyebrow">WVS Wave 7 · Predictive research</div>
              <h1>What makes a life<br><span class="accent">feel good?</span></h1>
              <p class="hero-deck">I began this research expecting economic wellbeing to be a key predictor of life satisfaction. The findings challenged that assumption.</p>
              <p class="hero-deck">Using World Values Survey Wave 7 data, I developed a machine learning model to explore how economic, social, and personal factors predict life satisfaction—and what lies beyond economic success.</p>
            </div></div>
            """,
            unsafe_allow_html=True,
        )
        page_navigation()

elif page == "Prediction":
    with st.container(key="prediction_header"):
        brand_line()
        st.markdown(
            "<div class='eyebrow'>Interactive exploration</div>"
            "<h1>Your wellbeing<br><span class='accent'>profile.</span></h1>",
            unsafe_allow_html=True,
        )
        st.write("Explore a predicted life-satisfaction score using a reduced set of WVS Wave 7 inputs.")
        page_navigation()

elif page == "Model Insights":
    with st.container(key="insights_header"):
        brand_line()
        st.markdown(
            "<div class='eyebrow'>Model interpretation</div>"
            "<h1>Patterns, not<br><span class='accent'>prescriptions.</span></h1>",
            unsafe_allow_html=True,
        )
        st.write("Explore how feature rankings compare across interpretation methods, and what selected patterns may invite us to consider. The full Random Forest feature-importance chart appears on the Home page. Predictive importance is not causal evidence.")
        page_navigation()

elif page == "Workflow":
    with st.container(key="workflow_header"):
        brand_line()
        st.markdown(
            "<div class='eyebrow'>Research process</div>"
            "<h1>From data to<br><span class='accent'>predictive insight.</span></h1>",
            unsafe_allow_html=True,
        )
        st.write("The study is organized into five layers and ten steps, from understanding the WVS questionnaire to model evaluation and reflection.")
        page_navigation()

if page == "Home":
    section_head("The study at a glance", "A global survey, one central question",
                "The analysis uses World Values Survey Wave 7 and examines self-reported life satisfaction across countries and regions.")
    overview = [
        ("WVS Wave 7", "Data source", "World Values Survey"),
        ("96,700", "Valid target responses", "Non-missing responses to Q49 · Life Satisfaction"),
        ("66", "Countries & regions", "Country/region entries represented in the dataset"),
        ("20", "Selected model inputs", "19 survey questions + Country/region"),
    ]
    overview_cols = st.columns(4)
    for col, (value, title, desc) in zip(overview_cols, overview):
        with col:
            st.markdown(
                f"<div class='metric-card'><div class='metric-value'>{value}</div>"
                f"<div class='metric-title'>{title}</div><div class='metric-desc'>{desc}</div></div>",
                unsafe_allow_html=True,
            )

    section_head("Top 20 feature ranking", "What stood out in the model",
                "The 20 highest-ranked features from the evaluated Top30 Random Forest, shown with aggregated feature-importance percentages and concise descriptions.")
    st.image(ASSET_DIR / "top20_importance_with_question_text.png", use_container_width=True)
    st.caption("Importance percentages come from the Notebook 05a final Top30 model. Country aggregates country/region indicators and is not a questionnaire item.")
    st.markdown(
        "<div class='soft-note'>Q50 · Financial Satisfaction accounts for 29.70% of aggregated importance, followed by Country (13.15%) and Q48 · Freedom of Choice and Control (11.43%). The Top 20 features account for 87.48% of total aggregated importance. Feature importance describes model reliance, not causal effects.</div>",
        unsafe_allow_html=True,
    )

    section_head("A finding to explore", "Financial satisfaction in the model",
                "A SHAP dependence plot from the Top20 feature experiment in Notebook 05a.")
    plot_col, explain_col = st.columns([1.25, 1])
    with plot_col:
        st.image(ASSET_DIR / "shap_q50_q288_dependence.png", use_container_width=True)
        st.caption("SHAP dependence plot · Q50 colored by Q288 · Notebook 05a")
    with explain_col:
        st.markdown(
            "<div class='content-card'><div class='section-label'>How to read it</div>"
            "<h3>Each dot is a respondent in the SHAP sample.</h3>"
            "<p style='color:#786B63;line-height:1.7'>The horizontal axis shows Q50 · Satisfaction with the financial situation of the household. The vertical axis shows Q50's SHAP value—its contribution to the model's prediction for that respondent.</p>"
            "<p style='color:#786B63;line-height:1.7'>Color represents Q288 · Household income scale (within the country): red indicates a higher reported income step and blue a lower one. Positive SHAP values contribute toward a higher model prediction; negative values contribute toward a lower prediction.</p>"
            "<p style='color:#786B63;line-height:1.7'>Some respondents at higher income steps (red) show negative SHAP values, while some at lower steps (blue) show positive values. This suggests that household income position and financial satisfaction do not always align in the same way with the model's predictions.</p>"
            "<div class='warm-note'>The pattern describes the model's predictive behavior. It does not show that financial satisfaction causes life satisfaction.</div></div>",
            unsafe_allow_html=True,
        )
    q1, q2, q3 = st.columns(3)
    with q1:
        reflection_card("REFLECTION 01", "Financial security",
                       "Is feeling financially secure as important as having a higher income?")
    with q2:
        reflection_card("REFLECTION 02", "Comparisons",
                       "Do people evaluate their financial wellbeing through their own circumstances or by comparing themselves with others?")
    with q3:
        reflection_card("REFLECTION 03", "Beyond economics",
                       "Can economic success fully capture how people experience wellbeing?")

elif page == "Prediction":
    st.markdown(
        "<div class='warm-note'><strong>Model choice:</strong> In the Notebook 05a feature-count experiments, the Top 30 Random Forest performed slightly better than Top 20 (5-fold CV R² = 0.4533 vs. 0.4460). This interactive version uses 20 model inputs for ease of completion: 19 survey questions plus Country/region. The reduced-input model is separate from the Top 30 model. To keep the downloadable artifact lighter, the App uses a 100-tree sub-ensemble from a 200-tree reduced-input fit. The Notebook Top20 CV score used fold-wise feature selection; these scores are context, not a direct evaluation of this exact fixed App input set.</div>",
        unsafe_allow_html=True,
    )
    st.caption("You may skip any survey response. Skipped items are imputed using the training-data hierarchical imputation procedure; more skipped answers mean the profile relies more on typical values.")
    bundle = load_app_model()
    country_codes = list(bundle["encoder"].categories_[0])
    step_labels = ["01 · Wellbeing", "02 · Society & outlook", "03 · Values & community"]
    # Use buttons rather than a radio widget for section navigation. A radio's
    # widget state can override question_step on the rerun after form submission,
    # preventing Save & Continue from advancing and making entered answers appear lost.
    current_step = int(st.session_state.get("question_step", 0))
    nav_cols = st.columns(3)
    for nav_idx, (nav_col, nav_label) in enumerate(zip(nav_cols, step_labels)):
        with nav_col:
            if st.button(nav_label, key=f"question_step_nav_{nav_idx}",
                         type="primary" if nav_idx == current_step else "secondary",
                         use_container_width=True):
                st.session_state.question_step = nav_idx
                st.rerun()
    idx = int(st.session_state.get("question_step", 0))
    st.progress((idx + 1) / 3, text=f"Section {idx + 1} of 3")
    number, title, features, description = QUESTION_GROUPS[idx]
    section_head(f"Section {number}", title, description)
    st.caption(f"{len(features)} survey questions in this section")
    responses = st.session_state.get("wvs_responses", {})
    with st.form(f"prediction_form_{idx}"):
        if idx == 0:
            st.caption("Required for prediction: choose one of the 66 WVS Wave 7 countries/regions represented in the model training data. The model cannot generate a country-specific estimate for a location outside this list.")
            country_code = st.selectbox(
                "Country / region (required)",
                country_codes,
                index=None,
                placeholder="Select a country or region…",
                format_func=country_label,
                key="wvs_country_input",
                help="Only the 66 country/region categories included in WVS Wave 7 model training are supported.",
            )
        else:
            # Widget keys are cleaned up by Streamlit when a widget is not
            # rendered on the current rerun. Keep the submitted country in a
            # separate non-widget session key so it survives section changes.
            country_code = st.session_state.get("wvs_country_saved")
            if country_code is None:
                st.warning("Please return to Section 1 and select a country/region before continuing.")
            else:
                st.info(f"Country / region: {country_label(country_code)} (selected in Section 1)")
        cols = st.columns(2)
        for i, feature in enumerate(features):
            short, prompt, scale, valid_values, endpoint_labels = QUESTION_INFO[feature]
            with cols[i % 2]:
                st.markdown(f"<div class='feature-card'><div class='feature-code'>{feature} · {short}</div><div class='feature-text'>{prompt}</div></div>", unsafe_allow_html=True)
                options = [None] + valid_values
                def _format_option(value, endpoints=endpoint_labels):
                    if value is None: return "Skip / prefer not to answer"
                    label = endpoints.get(value)
                    return f"{value} · {label}" if label else str(value)
                selected = st.selectbox(f"Response · {feature}", options, index=0, format_func=_format_option, key=f"answer_{feature}", help=scale)
                responses[feature] = selected
        submitted = st.form_submit_button("Predict life satisfaction" if idx == 2 else "Save responses & continue →", type="primary")
    st.session_state["wvs_responses"] = responses
    if submitted:
        # On Section 1, copy the form widget value into a persistent session
        # key before changing sections. Streamlit may discard the widget key
        # once its selectbox is no longer rendered.
        if idx == 0:
            selected_country = st.session_state.get("wvs_country_input")
            if selected_country is not None:
                st.session_state["wvs_country_saved"] = selected_country
        else:
            selected_country = st.session_state.get("wvs_country_saved")
        if selected_country is None:
            st.warning("Please select one of the supported 66 countries/regions in Section 1. Country/region is required for prediction.")
        elif idx < 2:
            st.session_state.question_step = idx + 1
            st.rerun()
        else:
            user_row = {feature: responses.get(feature) for feature in bundle["features"]}
            user_row["Country"] = selected_country
            imputed = apply_user_imputation(user_row, bundle)
            survey_values = np.array([float(imputed[f]) for f in bundle["features"]], dtype=float).reshape(1, -1)
            country_values = bundle["encoder"].transform(pd.DataFrame({"Country": [user_row["Country"]]}))
            model_input = np.hstack([survey_values, country_values])
            prediction = float(bundle["model"].predict(model_input)[0])
            st.session_state["wvs_prediction"] = prediction
            st.session_state["wvs_prediction_ready"] = True
            st.rerun()
    if st.session_state.get("wvs_prediction_ready"):
        prediction = st.session_state.get("wvs_prediction")
        st.markdown(
            f"<div class='content-card'><div class='section-label'>Your model estimate</div><h2 style='margin:.2rem 0 1rem'>{prediction:.1f} <span style='font-size:1rem;color:#786B63'>/ 10</span></h2><p style='color:#786B63;line-height:1.7'>This is a statistical estimate based on patterns in WVS Wave 7, not a definitive assessment of an individual’s wellbeing. The model identifies predictive associations, not causes.</p></div>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(prediction / 10, 0.0), 1.0))
    with st.expander("About this reduced-input model"):
        metrics = bundle["app_model_test_metrics"]
        st.write(f"The App’s 100-tree reduced-input sub-ensemble was evaluated on the provided test split: R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}.")
        st.caption("The test split was used in earlier project analyses, so these results are supplementary rather than a fresh independent evaluation. The 5-fold CV comparison above is the basis for noting that Top 30 performed better than Top 20 in the Notebook 05a feature-count experiment.")

elif page == "Model Insights":

    section_head("Interpretation methods", "Do the rankings agree?",
                "Compare impurity-based Random Forest importance, permutation importance, and mean absolute SHAP rankings.")
    st.image(ASSET_DIR / "ranking_across_methods.png", use_container_width=True)
    st.caption("Feature Ranking Across Three Interpretation Methods · rank 1 indicates the highest importance within each method.")
    st.markdown(
        "<div class='content-card'><h3>A stable core, with some variation</h3>"
        "<p style='color:#786B63;line-height:1.7'>Q50 remains ranked first across all three methods. Q48 and Q47 also appear near the top across methods. Some other features move more substantially between methods, reminding us that feature rankings depend partly on how importance is measured.</p></div>",
        unsafe_allow_html=True,
    )

    section_head("Freedom of choice and control", "Q48 · Freedom of Choice and Control",
                "Some people feel they have completely free choice and control over their lives, while others feel that what they do has little effect on what happens to them.")
    st.markdown(
        "<div class='content-card'><p style='color:#786B63;line-height:1.7'>Q48 asks respondents to rate how much freedom of choice and control they feel they have over the way their life turns out, on a 1–10 scale. It ranked among the strongest predictors in the model.</p>"
        "<p><strong>Questions to consider</strong></p><ul><li>Can material success bring life satisfaction when people have little control over their choices?</li>"
        "<li>Does wellbeing depend more on the opportunities people have, or on the opportunities they believe they have?</li></ul></div>",
        unsafe_allow_html=True,
    )

    section_head("Country & social context", "Life satisfaction across 66 countries and regions",
                "Country was an important predictor in the final model. The chart shows differences in average self-reported life satisfaction across the WVS Wave 7 country/region entries, labeled with country and region names.")
    st.image(ASSET_DIR / "country_life_satisfaction.png", use_container_width=True)
    st.caption("Average life satisfaction by country/region, shown with country and region names rather than three-letter codes. The dashed line marks the overall respondent-level mean (7.06).")
    st.markdown(
        "<div class='content-card'><p style='color:#786B63;line-height:1.7'>Country-level differences may reflect many contextual and measurement factors. This chart is descriptive and does not identify which national conditions explain the differences.</p>"
        "<p><strong>Questions to consider</strong></p><ul><li>Why do average life satisfaction levels differ across countries—and how much might these differences reflect social conditions, cultural contexts, or response patterns?</li>"
        "<li>Can national-level indicators help explain wellbeing without overlooking differences among people within the same country?</li></ul></div>",
        unsafe_allow_html=True,
    )

    section_head("Other predictors & future research", "Health and personal beliefs",
                "Two additional features invite questions about wellbeing beyond economic circumstances.")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "<div class='reflection-card'><div class='reflection-num'>Q47</div><div class='reflection-title'>Self-rated Health</div>"
            "<div class='reflection-question'>The WVS asks respondents how they would describe their state of health these days, from very good to very poor.</div>"
            "<p style='margin-top:1rem'><strong>Question to consider</strong></p><div class='reflection-question'>How might self-rated health relate to life satisfaction beyond financial circumstances?</div></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            "<div class='reflection-card'><div class='reflection-num'>Q164</div><div class='reflection-title'>Importance of God</div>"
            "<div class='reflection-question'>The WVS asks respondents how important God is in their life, on a 1–10 scale.</div>"
            "<p style='margin-top:1rem'><strong>Question to consider</strong></p><div class='reflection-question'>Does the importance of personal beliefs vary across countries and social contexts?</div></div>",
            unsafe_allow_html=True,
        )

    section_head("Key methods", "The model toolkit")
    m1, m2 = st.columns([1, 1])
    with m1:
        st.markdown(
            "<div class='content-card'><div class='method-item'><div class='method-icon'>✦</div><div><div class='method-name'>Data cleaning & imputation</div><div class='method-detail'>Special-code handling and hierarchical imputation.</div></div></div>"
            "<div class='method-item'><div class='method-icon'>⌁</div><div><div class='method-name'>Feature selection</div><div class='method-detail'>Assessment and comparison of selected survey feature sets.</div></div></div>"
            "<div class='method-item'><div class='method-icon'>▥</div><div><div class='method-name'>Model comparison</div><div class='method-detail'>Linear Regression, Decision Tree and Random Forest.</div></div></div></div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            "<div class='content-card'><div class='method-item'><div class='method-icon'>↔</div><div><div class='method-name'>Cross-validation</div><div class='method-detail'>Compare predictive performance across validation folds.</div></div></div>"
            "<div class='method-item'><div class='method-icon'>⌖</div><div><div class='method-name'>Model interpretation</div><div class='method-detail'>Feature Importance, Permutation Importance and SHAP.</div></div></div>"
            "<div class='soft-note' style='margin-top:1rem'>The interactive Prediction page uses a separately trained reduced-input Random Forest (19 survey questions + Country). The strongest evaluated configuration in Notebook 05a remains Top30.</div></div>",
            unsafe_allow_html=True,
        )

    section_head("Model performance", "Top 30 performed slightly better")
    st.markdown(
        "<div class='content-card'><p style='color:#786B63;line-height:1.7'>In the Notebook 05a feature-count experiments, the Top 30 Random Forest achieved slightly stronger 5-fold cross-validation performance than Top 20. The Top 30 configuration was retained as the best-performing evaluated model.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Top 30 · Random Forest · Hierarchical Imputation**")
    a, b, c = st.columns(3)
    a.metric("Mean CV R²", "0.4533")
    b.metric("Mean CV MAE", "1.1934")
    c.metric("Mean CV RMSE", "1.6569")
    st.markdown("**Top 20 · Random Forest · Hierarchical Imputation**")
    a, b, c = st.columns(3)
    a.metric("Mean CV R²", "0.4460")
    b.metric("Mean CV MAE", "1.1980")
    c.metric("Mean CV RMSE", "1.6679")
    st.caption("Notebook 05a feature-count experiments. Top20 CV used fold-wise feature selection; the App uses a fixed reduced input set for usability, retrained as a separate model. These CV scores are contextual and not a direct evaluation of the exact App input set.")
    st.markdown(
        "<div class='soft-note'>R² summarizes predictive performance relative to variation in the target. MAE and RMSE describe error on the 1–10 life-satisfaction scale; RMSE penalizes larger errors more strongly.</div>",
        unsafe_allow_html=True,
    )

elif page == "Workflow":
    section_head("Research workflow", "10 steps · 5 layers")
    st.image(ASSET_DIR / "research_workflow_warm_sun.png", use_container_width=True)
    st.caption("Workflow diagram based on the original project workflow. The original structure and step icons are retained, with the blue accents shifted to a warm terracotta palette.")

    section_head("Notebook sequence", "The research notebooks")
    notebooks = [
        "00 Overview", "01 Inventory", "02 Special Codes", "03 Feature Discovery",
        "04 EDA", "04a Correlation", "05 Feature Engineering", "05a Experiments", "06 Final Results",
    ]
    st.markdown("<div class='notebook-wrap'>" + "".join(f"<span class='notebook-chip'>{n}</span>" for n in notebooks) + "</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='warm-note' style='margin-top:1.2rem'>The workflow describes the research process. The model configuration used by the App is separately identified on Model Insights, because the report evaluation model and planned App model are not identical versions.</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='footer'>BEYOND ECONOMIC SUCCESS&nbsp;&nbsp; · &nbsp;&nbsp;WVS WAVE 7&nbsp;&nbsp; · &nbsp;&nbsp;RESEARCH PROTOTYPE</div>", unsafe_allow_html=True)
