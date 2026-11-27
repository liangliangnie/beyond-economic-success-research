from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import base64
import html
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

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
    """Fill skipped survey responses using global medians; Country is not used."""
    result = {feature: row.get(feature) for feature in bundle["features"]}
    medians = bundle["imputer_medians"]
    for feature in bundle["features"]:
        value = result.get(feature)
        if value is None or pd.isna(value):
            result[feature] = medians[feature]
    return result


def load_chart_data(filename):
    return pd.read_csv(ASSET_DIR / filename)


def plotly_warm(fig, height=430):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFEFC",
        font=dict(family="DM Sans, sans-serif", color="#2D211D"),
        margin=dict(l=18, r=18, t=35, b=25),
        colorway=["#D95B18", "#7D927B", "#D9A06A", "#8C7568", "#C98263"],
        xaxis=dict(gridcolor="#F0E7DE", zerolinecolor="#D8C9BC"),
        yaxis=dict(gridcolor="#F0E7DE", zerolinecolor="#D8C9BC"),
        legend=dict(bgcolor="rgba(255,254,252,.7)")
    )
    return fig


def show_plotly(fig, height=430):
    fig = plotly_warm(fig, height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "responsive": True})


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
    .st-key-prediction_header, .st-key-insights_header, .st-key-extra_insights_header, .st-key-workflow_header { margin-bottom:1.4rem; }
    .st-key-home_header div[data-testid="stRadio"],
    .st-key-prediction_header div[data-testid="stRadio"],
    .st-key-insights_header div[data-testid="stRadio"],
    .st-key-extra_insights_header div[data-testid="stRadio"],
    .st-key-workflow_header div[data-testid="stRadio"] { position:relative; z-index:2; margin-top:1.15rem; }
    .st-key-home_header div[role="radiogroup"],
    .st-key-prediction_header div[role="radiogroup"],
    .st-key-insights_header div[role="radiogroup"],
    .st-key-extra_insights_header div[role="radiogroup"],
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



def show_html_figure(image_path, caption=None, height=460):
    """Responsive HTML image frame for source figures whose underlying arrays were not retained."""
    image_path = Path(image_path)
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    caption_html = f"<p class='fig-caption'>{html.escape(caption)}</p>" if caption else ""
    markup = f"""<div style='background:#FFFEFC;border:1px solid #E8DCD0;border-radius:20px;padding:12px;overflow:hidden'>
      <img src='data:{mime};base64,{encoded}' alt='Research figure' style='display:block;width:100%;height:auto;max-height:780px;object-fit:contain;border-radius:12px'>
      {caption_html}</div><style>.fig-caption{{font:12px 'DM Sans',sans-serif;color:#786B63;margin:8px 4px 2px;line-height:1.5}}</style>"""
    components.html(markup, height=height, scrolling=True)


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


def show_zoomable_image(image_path, caption=None, component_height=470):
    # Responsive preview with an in-frame modal, zoom controls, pinch and pan.
    image_path = Path(image_path)
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    data_url = f"data:{mime};base64,{encoded}"
    caption_html = f"<div class='zoom-caption'>{html.escape(caption)}</div>" if caption else ""
    component_html = f'''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=5,user-scalable=yes"><style>
    *{{box-sizing:border-box}} body{{margin:0;font-family:Arial,sans-serif;background:transparent;color:#786B63}}
    .preview{{width:100%;display:block;border-radius:12px;cursor:zoom-in;max-height:420px;object-fit:contain;background:#fff}}
    .hint{{font-size:12px;text-align:center;margin:7px 0 0;color:#8a7a70}}
    .zoom-caption{{font-size:12px;color:#786B63;margin-top:7px;line-height:1.45}}
    .viewer{{display:none;position:fixed;inset:0;z-index:9999;background:rgba(35,25,20,.96);padding:42px 8px 10px;overflow:hidden;touch-action:none}}
    .viewer.open{{display:block}}
    .toolbar{{position:absolute;top:6px;left:8px;right:8px;display:flex;justify-content:flex-end;gap:7px;z-index:2}}
    button{{border:1px solid rgba(255,255,255,.35);background:#fff;color:#2D211D;border-radius:999px;padding:8px 12px;font-size:15px;font-weight:700;cursor:pointer}}
    .stage{{width:100%;height:100%;display:flex;align-items:center;justify-content:center;overflow:hidden}}
    .large{{max-width:100%;max-height:100%;object-fit:contain;transform-origin:center center;user-select:none;-webkit-user-drag:none;touch-action:none}}
    .modalhint{{position:absolute;bottom:8px;left:0;right:0;text-align:center;color:#fff;font-size:11px;opacity:.8;pointer-events:none}}
    </style></head><body>
    <img class="preview" src="{data_url}" alt="Chart preview" onclick="openViewer()">
    <div class="hint">Tap image to enlarge · pinch or use + / − to zoom</div>{caption_html}
    <div class="viewer" id="viewer" onclick="if(event.target===this)closeViewer()">
      <div class="toolbar"><button onclick="zoomBy(1.35);event.stopPropagation()">＋</button><button onclick="zoomBy(0.75);event.stopPropagation()">−</button><button onclick="resetZoom();event.stopPropagation()">Reset</button><button onclick="closeViewer();event.stopPropagation()">Close ✕</button></div>
      <div class="stage" id="stage"><img class="large" id="large" src="{data_url}" alt="Enlarged chart"></div>
      <div class="modalhint">Pinch to zoom · drag to move · tap outside or Close to return</div>
    </div>
    <script>
    const viewer=document.getElementById('viewer'),img=document.getElementById('large'),stage=document.getElementById('stage');
    let scale=1,tx=0,ty=0,startX=0,startY=0,startTx=0,startTy=0,startDist=0,startScale=1,drag=false;
    function apply(){{img.style.transform=`translate(${{tx}}px,${{ty}}px) scale(${{scale}})`;}}
    function openViewer(){{viewer.classList.add('open');resetZoom();}}
    function closeViewer(){{viewer.classList.remove('open');}}
    function resetZoom(){{scale=1;tx=0;ty=0;apply();}}
    function zoomBy(f){{scale=Math.max(1,Math.min(5,scale*f));apply();}}
    function dist(a,b){{return Math.hypot(a.clientX-b.clientX,a.clientY-b.clientY);}}
    stage.addEventListener('touchstart',e=>{{e.preventDefault();if(e.touches.length===2){{startDist=dist(e.touches[0],e.touches[1]);startScale=scale;}}else if(e.touches.length===1){{drag=true;startX=e.touches[0].clientX;startY=e.touches[0].clientY;startTx=tx;startTy=ty;}}}},{{passive:false}});
    stage.addEventListener('touchmove',e=>{{e.preventDefault();if(e.touches.length===2&&startDist){{scale=Math.max(1,Math.min(5,startScale*dist(e.touches[0],e.touches[1])/startDist));apply();}}else if(e.touches.length===1&&drag&&scale>1){{tx=startTx+e.touches[0].clientX-startX;ty=startTy+e.touches[0].clientY-startY;apply();}}}},{{passive:false}});
    stage.addEventListener('touchend',()=>{{drag=false;startDist=0;}});
    stage.addEventListener('mousedown',e=>{{if(scale>1){{drag=true;startX=e.clientX;startY=e.clientY;startTx=tx;startTy=ty;}}}});
    window.addEventListener('mousemove',e=>{{if(drag){{tx=startTx+e.clientX-startX;ty=startTy+e.clientY-startY;apply();}}}});
    window.addEventListener('mouseup',()=>drag=false);
    viewer.addEventListener('wheel',e=>{{e.preventDefault();zoomBy(e.deltaY<0?1.12:.89);}},{{passive:false}});
    document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeViewer();if(e.key==='+'||e.key==='=')zoomBy(1.2);if(e.key==='-')zoomBy(.8);}});
    </script></body></html>'''
    components.html(component_html, height=component_height, scrolling=False)


PAGES = ["Home", "Prediction", "Model Insights", "Extra Insights", "Workflow"]
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


FEATURE_IMPORTANCE = [
    ("Q50 · Household financial satisfaction", 29.70),
    ("Country · Country / region (aggregated)", 13.15),
    ("Q48 · Freedom of choice and control", 11.43),
    ("Q47 · Self-rated health", 2.69),
    ("Q251 · Perceived democratic governance", 2.56),
    ("Q252 · Satisfaction with political system", 2.55),
    ("Q288 · Household income position", 2.50),
    ("Q163 · Benefits of science and technology", 2.44),
    ("Q106 · Income equality vs. income differences", 2.37),
    ("Q158 · Science and technology improve daily life", 2.11),
    ("Q159 · Science creates opportunities for next generation", 2.09),
    ("Q249 · Women’s equal rights as a democratic feature", 1.91),
    ("Q250 · Importance of living in a democracy", 1.80),
    ("Q164 · Importance of God", 1.72),
    ("Q287 · Subjective social class", 1.53),
    ("Q131 · Security in the neighborhood", 1.45),
    ("Q080 · Confidence in women’s organizations", 1.40),
    ("Q253 · Respect for individual human rights", 1.37),
    ("Q075 · Confidence in universities", 1.37),
    ("Q088 · Confidence in the World Health Organization", 1.34),
]

def show_feature_importance_html():
    rows = []
    for idx, (label, value) in enumerate(FEATURE_IMPORTANCE, start=1):
        safe_label = html.escape(label)
        bar_class = "primary" if idx <= 3 else "secondary"
        rows.append(
            f"<div class='fi-row'><div class='fi-rank'>{idx:02d}</div>"
            f"<div class='fi-label'>{safe_label}</div>"
            f"<div class='fi-track'><div class='fi-bar {bar_class}' style='width:{value/30*100:.3f}%'></div></div>"
            f"<div class='fi-value'>{value:.2f}%</div></div>"
        )
    st.markdown(
        "<style>"
        ".fi-card{background:#FFFCF8;border:1px solid #EADBCB;border-radius:18px;padding:18px 16px;overflow:hidden}"
        ".fi-row{display:grid;grid-template-columns:28px minmax(185px,1.2fr) minmax(100px,1.5fr) 52px;gap:10px;align-items:center;margin:8px 0}"
        ".fi-rank{font-size:11px;font-weight:800;color:#A18D7E}"
        ".fi-label{font-size:12px;line-height:1.35;color:#3B2D27}"
        ".fi-track{height:14px;background:#F1E7DD;border-radius:999px;overflow:hidden}"
        ".fi-bar{height:100%;border-radius:999px}"
        ".fi-bar.primary{background:#D95B12}.fi-bar.secondary{background:#A6B59E}"
        ".fi-value{font-size:12px;font-weight:700;color:#4A3931;text-align:right;font-variant-numeric:tabular-nums}"
        ".fi-axis{display:flex;justify-content:space-between;margin:10px 62px 0 230px;color:#8A786B;font-size:10px}"
        ".fi-foot{font-size:12px;color:#786B63;line-height:1.55;margin-top:14px}"
        "@media(max-width:720px){.fi-row{grid-template-columns:24px minmax(0,1fr) 48px;gap:7px}.fi-track{grid-column:2;grid-row:2}.fi-value{grid-column:3;grid-row:1/3}.fi-label{font-size:11px}.fi-axis{margin:8px 55px 0 32px}}"
        "</style>"
        "<div class='fi-card'>" + "".join(rows) +
        "<div class='fi-axis'><span>0%</span><span>10%</span><span>20%</span><span>30%</span></div>"
        "<div class='fi-foot'><strong>The Top 20 features account for 87.48% of total aggregated importance.</strong><br>Country combines country/region indicator variables. Importance reflects model reliance, not causal effects.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

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

elif page == "Extra Insights":
    with st.container(key="extra_insights_header"):
        brand_line()
        st.markdown(
            "<div class='eyebrow'>Exploratory analysis</div>"
            "<h1>Patterns beyond<br><span class='accent'>the model.</span></h1>",
            unsafe_allow_html=True,
        )
        st.write("Explore age and gender patterns in WVS Wave 7 beyond the predictive machine learning analysis.")
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
        ("~97,220", "Respondent records", "Original respondent records in the dataset"),
        ("613", "Original variables", "Variables in the original dataset"),
        ("66", "Countries & regions", "Country/region entries represented in the dataset"),
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
                "The 20 highest-ranked features in the feature-importance analysis, shown as aggregated percentages.")
    show_feature_importance_html()
    st.caption("Higher bars show features the research model relied on more; Country appears in the research ranking but is not an input in the interactive prediction survey.")

    section_head("A finding to explore", "Financial satisfaction in the model",
                "A SHAP dependence plot from the Top20 feature experiment in Notebook 05a.")
    plot_col, explain_col = st.columns([1.25, 1])
    with plot_col:
        show_html_figure(ASSET_DIR / "shap_q50_q288_dependence.png", "SHAP dependence plot", height=500)
        st.caption("Each dot is a respondent: see how financial satisfaction contributes to the model’s prediction, with color showing income position.")
    with explain_col:
        st.markdown(
            "<div class='content-card'><div class='section-label'>How to read it</div>"
            "<p style='color:#786B63;line-height:1.6'>Each dot is a respondent. The x-axis is financial satisfaction; the y-axis shows how it shifts the model’s prediction. Positive SHAP values push the prediction higher, negative values lower. Color shows household income position.</p>"
            "<div class='warm-note'>SHAP explains the model’s prediction; it does not show cause and effect.</div></div>",
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
        "<div class='warm-note'><strong>Model choice:</strong> In the Notebook 05a feature-count experiments, the Top 30 Random Forest performed slightly better than Top 20 (5-fold CV R² = 0.4533 vs. 0.4460). The interactive prediction model was retrained with 19 survey questions and no Country input. Skipped responses are filled using global median values. The Notebook Top20/Top30 experiments are separate research models; their scores are context rather than direct evaluations of this app model.</div>",
        unsafe_allow_html=True,
    )
    st.caption("You may skip any survey response. Skipped items are imputed using global median values from valid WVS responses; more skipped answers mean the profile relies more on typical values.")
    bundle = load_app_model()
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
            st.caption("Country is not required. The prediction uses your survey responses only, so you can complete it regardless of where you live.")
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
        if idx < 2:
            st.session_state.question_step = idx + 1
            st.rerun()
        else:
            user_row = {feature: responses.get(feature) for feature in bundle["features"]}
            imputed = apply_user_imputation(user_row, bundle)
            model_input = np.array([float(imputed[f]) for f in bundle["features"]], dtype=float).reshape(1, -1)
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
        st.write(f"The 100-tree app model without Country was evaluated on the provided test split: R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}.")
        st.caption("Removing Country reduced predictive performance compared with the previous app model that included Country (previous test R² ≈ 0.450; MAE ≈ 1.186). The current test split was used in earlier project analyses, so these results are indicative rather than a fresh independent evaluation.")
        st.caption("The 5-fold CV comparison above refers to the separate Notebook 05a feature-count experiments, not this no-Country app model.")

elif page == "Model Insights":

    section_head("Interpretation methods", "Do the rankings agree?",
                "Compare impurity-based Random Forest importance, permutation importance, and mean absolute SHAP rankings.")
    show_html_figure(ASSET_DIR / "ranking_across_methods.png", "Feature rankings across interpretation methods", height=500)
    st.caption("Each method ranks features by importance; rank 1 is the highest within that method.")
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
                "Country appeared as an important predictor in the original research model, but it is intentionally excluded from the interactive survey. This chart shows weighted differences in average self-reported life satisfaction across WVS Wave 7 country/region entries.")
    country_df = load_chart_data("country_summary_data.csv").sort_values("mean_q49", ascending=True)
    country_fig = px.bar(country_df, x="mean_q49", y="country", orientation="h", hover_data={"code": True, "n": True, "mean_q49": ":.2f"}, labels={"mean_q49":"Weighted mean life satisfaction (Q49)","country":"Country / region"})
    country_fig.add_vline(x=7.06, line_dash="dash", line_color="#8C7568", annotation_text="Overall mean 7.06")
    show_plotly(country_fig, height=920)
    st.caption("Weighted country/region averages; hover over bars for the country code and response count.")
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
            "<div class='content-card'><div class='method-item'><div class='method-icon'>✦</div><div><div class='method-name'>Data cleaning & imputation</div><div class='method-detail'>Special-code handling; app predictions use global median imputation.</div></div></div>"
            "<div class='method-item'><div class='method-icon'>⌁</div><div><div class='method-name'>Feature selection</div><div class='method-detail'>Assessment and comparison of selected survey feature sets.</div></div></div>"
            "<div class='method-item'><div class='method-icon'>▥</div><div><div class='method-name'>Model comparison</div><div class='method-detail'>Linear Regression, Decision Tree and Random Forest.</div></div></div></div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            "<div class='content-card'><div class='method-item'><div class='method-icon'>↔</div><div><div class='method-name'>Cross-validation</div><div class='method-detail'>Compare predictive performance across validation folds.</div></div></div>"
            "<div class='method-item'><div class='method-icon'>⌖</div><div><div class='method-name'>Model interpretation</div><div class='method-detail'>Feature Importance, Permutation Importance and SHAP.</div></div></div>"
            "<div class='soft-note' style='margin-top:1rem'>The interactive Prediction page uses a separately trained Random Forest with 19 survey questions and no Country input. The strongest evaluated configuration in Notebook 05a remains Top30.</div></div>",
            unsafe_allow_html=True,
        )

    section_head("Model performance", "Top 30 performed slightly better", "These scores summarize how closely predictions matched reported life satisfaction in cross-validation.")
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
    st.caption("Cross-validation compares the Top 30 and Top 20 experiments; the app uses a separate reduced-input model.")
    st.markdown(
        "<div class='soft-note'>R² summarizes predictive performance relative to variation in the target. MAE and RMSE describe error on the 1–10 life-satisfaction scale; RMSE penalizes larger errors more strongly.</div>",
        unsafe_allow_html=True,
    )

elif page == "Extra Insights":
    section_head("Beyond prediction", "Exploring patterns beyond the predictive model",
                 "The main analysis used machine learning to explore how economic, social, and personal factors predict life satisfaction. These additional analyses are descriptive and do not train a new machine learning model.")
    section_head("01 · Age patterns", "Does life satisfaction change with age?",
                 "Weighted mean responses across age groups for life satisfaction (Q49), financial satisfaction (Q50), and perceived freedom and control (Q48).")
    show_zoomable_image(ASSET_DIR / "age_patterns_three_outcomes.png", "Age patterns across three WVS self-reported outcomes", component_height=500)
    st.caption("Weighted average responses by age group for life satisfaction, financial satisfaction, and perceived freedom.")
    st.markdown(
        "<div class='content-card'><p style='color:#786B63;line-height:1.7'>Life satisfaction and financial satisfaction show relatively lower means around midlife and higher means among older respondents. Perceived freedom and control follows a different pattern, suggesting that dimensions of subjective wellbeing may not move together across age groups.</p><div class='warm-note'>This raises a question: is the relationship between age and life satisfaction more complex than a simple linear trend?</div></div>",
        unsafe_allow_html=True,
    )

    section_head("02 · Gender differences", "Does the gender gap depend on age?",
                 "The chart shows the female-minus-male difference in weighted mean Q49 within each age group.")
    show_zoomable_image(ASSET_DIR / "gender_gap_by_age.png", "Female minus male weighted mean life satisfaction by age group", component_height=450)
    st.caption("Positive values mean women report a higher average; negative values mean men do.")
    st.markdown(
        "<div class='content-card'><p style='color:#786B63;line-height:1.7'>The overall difference in life satisfaction between women and men is small. However, the direction of the difference varies across age groups. Women report slightly higher average life satisfaction in several younger and middle-age groups, while men report higher averages in the older groups.</p><div class='warm-note'>Rather than suggesting a universal gender difference, this pattern points to possible variation across different stages of life.</div></div>",
        unsafe_allow_html=True,
    )

    section_head("03 · Across countries", "Are gender differences consistent across countries?",
                 "Each bar represents the difference between women’s and men’s weighted mean Q49 within a country or region.")
    show_zoomable_image(ASSET_DIR / "country_gender_gap_full_names.png", "Country-level gender gap in life satisfaction; positive values indicate higher female means", component_height=700)
    st.caption("Each bar shows the within-country female-minus-male difference in weighted life-satisfaction averages.")
    st.markdown(
        "<div class='content-card'><p style='color:#786B63;line-height:1.7'>The gender gap is not uniform across countries and regions. In some places, women report higher average life satisfaction; in others, men report higher averages. A pooled global average can therefore conceal meaningful differences between country-level patterns.</p><div class='warm-note'>Positive values indicate a higher female mean; negative values indicate a higher male mean. The chart shows within-country gender gaps, not a ranking of countries by life satisfaction.</div></div>",
        unsafe_allow_html=True,
    )

    section_head("Further reflections", "What might these patterns invite us to ask?")
    r1, r2, r3 = st.columns(3)
    with r1:
        reflection_card("QUESTION 01", "Age and context", "How much of the age pattern remains after accounting for country differences?")
    with r2:
        reflection_card("QUESTION 02", "Different dimensions", "Do social support, financial security, and perceived freedom relate to life satisfaction differently across age groups?")
    with r3:
        reflection_card("QUESTION 03", "Cross-national variation", "To what extent might cultural and institutional contexts shape gender differences in subjective wellbeing?")
    st.markdown(
        "<div class='soft-note' style='margin-top:1.1rem'>These are exploratory, weighted descriptive analyses of cross-sectional survey responses. They identify patterns and associations, not causal effects, and are separate from the predictive machine learning model used in the main project.</div>",
        unsafe_allow_html=True,
    )

elif page == "Workflow":
    section_head("Research workflow", "10 steps · 5 layers")
    show_zoomable_image(ASSET_DIR / "research_workflow_warm_sun.png", "Research workflow")
    st.caption("The project workflow, from survey preparation to model evaluation and reflection.")

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
