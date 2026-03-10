import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary:    #0a0e1a;
    --bg-secondary:  #0f1628;
    --bg-card:       #131c30;
    --accent-cyan:   #00d4ff;
    --accent-green:  #00ff88;
    --accent-red:    #ff4757;
    --accent-orange: #ffa502;
    --accent-yellow: #ffdd59;
    --text-primary:  #e8eaf0;
    --text-secondary:#7a8299;
    --text-muted:    #4a5270;
    --border:        #1e2a42;
    --border-bright: #2a3a5c;
}

html, body, .stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}
.main .block-container {
    background: transparent !important;
    padding: 1.5rem 2rem 3rem 2rem;
    max-width: 1400px;
}
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-bright);
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; color: var(--text-primary) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary); border-radius: 12px;
    padding: 4px; border: 1px solid var(--border); gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 8px;
    color: var(--text-secondary) !important;
    font-family: 'Space Mono', monospace; font-size: 0.85rem;
    padding: 8px 20px; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-cyan) !important;
    color: var(--bg-primary) !important;
}
.stTabs [data-baseweb="tab-panel"] { background: transparent; padding-top: 1.5rem; }

/* Inputs */
.stTextInput input, .stSelectbox select, .stNumberInput input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    color: var(--text-primary) !important; border-radius: 8px !important;
}
.stTextInput input:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}
.stTextInput label, .stSelectbox label, .stRadio label, .stCheckbox label {
    color: var(--text-secondary) !important; font-size: 0.82rem !important;
    font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.04em;
}
div[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 8px !important; color: var(--text-primary) !important;
}
.stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    color: var(--text-primary) !important; border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.82rem !important;
}

/* Buttons */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent-cyan), #0088cc) !important;
    color: var(--bg-primary) !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important; font-size: 0.9rem !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important; letter-spacing: 0.05em;
}
.stButton button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(0,212,255,0.35) !important;
}
.stButton button[kind="secondary"] {
    background: var(--bg-card) !important; color: var(--text-primary) !important;
    border: 1px solid var(--border-bright) !important; border-radius: 10px !important;
}
.stDownloadButton button {
    background: var(--bg-card) !important; color: var(--accent-cyan) !important;
    border: 1px solid var(--accent-cyan) !important; border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.8rem !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; color: var(--text-primary) !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.85rem !important;
}
.streamlit-expanderContent {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important;
}

/* Alerts */
.stSuccess { background: rgba(0,255,136,0.08) !important; border: 1px solid rgba(0,255,136,0.3) !important; border-radius: 10px !important; color: var(--accent-green) !important; }
.stError   { background: rgba(255,71,87,0.08) !important;  border: 1px solid rgba(255,71,87,0.3) !important;  border-radius: 10px !important; color: var(--accent-red) !important; }
.stWarning { background: rgba(255,165,2,0.08) !important;  border: 1px solid rgba(255,165,2,0.3) !important;  border-radius: 10px !important; color: var(--accent-orange) !important; }
.stInfo    { background: rgba(0,212,255,0.08) !important;  border: 1px solid rgba(0,212,255,0.3) !important;  border-radius: 10px !important; color: var(--accent-cyan) !important; }

/* Metrics */
[data-testid="metric-container"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important; padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label {
    color: var(--text-secondary) !important; font-size: 0.78rem !important;
    text-transform: uppercase; letter-spacing: 0.06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent-cyan) !important;
    font-family: 'Space Mono', monospace !important; font-size: 1.5rem !important;
}

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
.stSpinner > div { border-top-color: var(--accent-cyan) !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

for key, default in [
    ('report_history', []),
    ('latest_report',  None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# HARDCODED MODEL RESULTS  (actual values from your notebook)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_RESULTS = {
    "Decision Tree":      {"Accuracy": 0.9171, "Precision": 0.9180, "Recall": 0.9099, "F1-Score": 0.9139, "ROC-AUC": 0.9219},
    "Random Forest":      {"Accuracy": 0.9419, "Precision": 0.9368, "Recall": 0.9435, "F1-Score": 0.9401, "ROC-AUC": 0.9881},
    "KNN":                {"Accuracy": 0.9188, "Precision": 0.9213, "Recall": 0.9099, "F1-Score": 0.9156, "ROC-AUC": 0.9723},
    "Naive Bayes":        {"Accuracy": 0.6547, "Precision": 0.9939, "Recall": 0.2880, "F1-Score": 0.4466, "ROC-AUC": 0.9635},
    "SVM":                {"Accuracy": 0.9402, "Precision": 0.9247, "Recall": 0.9541, "F1-Score": 0.9391, "ROC-AUC": 0.9827},
    "Logistic Regression":{"Accuracy": 0.9128, "Precision": 0.9000, "Recall": 0.9223, "F1-Score": 0.9110, "ROC-AUC": 0.9738},
}

TUNED_RESULTS = {
    "Decision Tree":  {"Accuracy": 0.9214, "F1-Score": 0.9178, "ROC-AUC": 0.9231, "Best Params": "max_depth=10, criterion=entropy"},
    "Random Forest":  {"Accuracy": 0.9453, "F1-Score": 0.9421, "ROC-AUC": 0.9889, "Best Params": "n_estimators=200, max_depth=None"},
    "KNN":            {"Accuracy": 0.9231, "F1-Score": 0.9196, "ROC-AUC": 0.9741, "Best Params": "n_neighbors=7, weights=distance"},
    "Naive Bayes":    {"Accuracy": 0.6547, "F1-Score": 0.4466, "ROC-AUC": 0.9635, "Best Params": "var_smoothing=1e-9"},
    "SVM":            {"Accuracy": 0.9419, "F1-Score": 0.9413, "ROC-AUC": 0.9841, "Best Params": "C=10, kernel=rbf, gamma=scale"},
}

BEST_MODEL_NAME = "SVM"
BEST_METRICS    = TUNED_RESULTS["SVM"]

EDA_FINDINGS = {
    "total_instances": 11055, "after_dedup": 5849, "duplicates_removed": 5206,
    "legitimate_pct": 51.62,  "phishing_pct": 48.38,
    "top_features": ["sslfinal_state","url_of_anchor","web_traffic","having_sub_domain","request_url"],
}

FEATURE_IMPORTANCE = {
    "sslfinal_state": 0.118, "url_of_anchor": 0.097, "web_traffic": 0.089,
    "having_sub_domain": 0.071, "request_url": 0.068, "page_rank": 0.062,
    "google_index": 0.055, "links_in_tags": 0.048, "age_of_domain": 0.046,
    "prefix_suffix": 0.041,
}

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = [
    "having_ip_address","url_length","shortining_service","having_at_symbol",
    "double_slash_redirecting","prefix_suffix","having_sub_domain","sslfinal_state",
    "domain_registration_length","favicon","port","https_token","request_url",
    "url_of_anchor","links_in_tags","sfh","submitting_to_email","abnormal_url",
    "redirect","on_mouseover","rightclick","popupwindow","iframe","age_of_domain",
    "dnsrecord","web_traffic","page_rank","google_index","links_pointing_to_page",
    "statistical_report",
]

FEATURE_LABELS = {
    "having_ip_address":         "Having IP Address",
    "url_length":                "URL Length",
    "shortining_service":        "Shortening Service",
    "having_at_symbol":          "Having @ Symbol",
    "double_slash_redirecting":  "Double Slash Redirecting",
    "prefix_suffix":             "Prefix/Suffix in Domain",
    "having_sub_domain":         "Having Sub Domain",
    "sslfinal_state":            "SSL Final State",
    "domain_registration_length":"Domain Registration Length",
    "favicon":                   "Favicon",
    "port":                      "Non-Standard Port",
    "https_token":               "HTTPS Token in URL",
    "request_url":               "Request URL",
    "url_of_anchor":             "URL of Anchor",
    "links_in_tags":             "Links in Tags",
    "sfh":                       "Server Form Handler",
    "submitting_to_email":       "Submitting to Email",
    "abnormal_url":              "Abnormal URL",
    "redirect":                  "Redirect Count",
    "on_mouseover":              "On Mouseover Changes Status",
    "rightclick":                "Right Click Disabled",
    "popupwindow":               "Pop-up Window",
    "iframe":                    "iFrame Redirection",
    "age_of_domain":             "Age of Domain",
    "dnsrecord":                 "DNS Record",
    "web_traffic":               "Web Traffic",
    "page_rank":                 "PageRank",
    "google_index":              "Google Index",
    "links_pointing_to_page":    "Links Pointing to Page",
    "statistical_report":        "Statistical Report",
}

OPTIONS_MAP = {-1: "Legitimate (-1)", 0: "Suspicious (0)", 1: "Phishing (1)"}

FEATURE_GROUPS = {
    "🌐 URL-Based":    ["having_ip_address","url_length","shortining_service","having_at_symbol","double_slash_redirecting","prefix_suffix","https_token"],
    "🔒 Domain & SSL": ["having_sub_domain","sslfinal_state","domain_registration_length","favicon","port","abnormal_url","age_of_domain","dnsrecord"],
    "📄 Page Content": ["request_url","url_of_anchor","links_in_tags","sfh","submitting_to_email","iframe","popupwindow"],
    "🖱️ Behavioral":   ["redirect","on_mouseover","rightclick"],
    "📊 Reputation":   ["web_traffic","page_rank","google_index","links_pointing_to_page","statistical_report"],
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD SAVED MODEL  (no notebook needed — just the 3 .joblib files)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    for mpath, spath, ipath in [
        ("best_model.joblib", "scaler.joblib",  "model_info.joblib"),
        ("best_model.pkl",    "scaler.pkl",      "model_info.pkl"),
    ]:
        if os.path.exists(mpath):
            m    = joblib.load(mpath)
            s    = joblib.load(spath)  if os.path.exists(spath)  else None
            info = joblib.load(ipath)  if os.path.exists(ipath)  else {
                "model_name":    BEST_MODEL_NAME,
                "needs_scaling": True,
            }
            return m, s, info
    return None, None, None

model, scaler, model_info = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI  — loaded from .streamlit/secrets.toml, never shown in the UI
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_gemini():
    try:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = os.environ.get("GEMINI_API_KEY", "")
        
        if not api_key:
            return None, "GEMINI_API_KEY not set"
        
        client = genai.Client(api_key=api_key)
        return client, None
    except Exception as e:
        return None, str(e)

gemini_client, gemini_error = load_gemini()

def generate_report(prompt: str) -> str:
    if gemini_client is None:
        return f"**Gemini unavailable:** {gemini_error}"
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"**Error generating report:** {str(e)}"

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(report_type: str, tone: str, sections: dict, language: str,
                 custom_override: str = "") -> str:
    if custom_override.strip():
        return custom_override

    active = [k for k, v in sections.items() if v]
    sections_str = ", ".join(active) if active else "all sections"

    other_str = "\n".join(
        f"  - {n}: Accuracy={m['Accuracy']:.4f}, F1={m['F1-Score']:.4f}, ROC-AUC={m['ROC-AUC']:.4f}"
        for n, m in MODEL_RESULTS.items() if n != BEST_MODEL_NAME
    )
    top_feats = ", ".join(f"{f} ({v:.3f})" for f, v in list(FEATURE_IMPORTANCE.items())[:7])

    tone_guide = {
        "Academic":  "Use formal academic language with technical terminology, passive voice where appropriate, and rigorous methodology references.",
        "Business":  "Use professional but accessible language. Focus on risk, business impact, and actionable insights. Avoid heavy jargon.",
        "Technical": "Use precise technical language for ML engineers. Include statistical details, methodology decisions, and implementation notes.",
    }
    type_guide = {
        "Executive Summary":       "Write a concise 1–2 paragraph summary. Focus on results and bottom-line impact only.",
        "Detailed Analysis":       "Write a comprehensive multi-section analysis with headers, bullet points, and structured narrative.",
        "Technical Documentation": "Write a thorough technical document covering methodology, validation, model selection rationale, and limitations.",
        "Business Presentation":   "Write stakeholder-ready content with clear headers, highlighted key numbers, and business value framing.",
    }
    section_guidance = {
        "Problem Statement":
            "Describe the phishing detection problem, its security and business stakes, and why ML is an appropriate approach.",
        "Dataset Description":
            f"UCI Phishing Websites dataset: {EDA_FINDINGS['total_instances']:,} instances "
            f"({EDA_FINDINGS['after_dedup']:,} after deduplication), 30 integer-encoded features (-1/0/1), "
            f"binary target. Sources: PhishTank, MillerSmiles, Google.",
        "EDA Findings":
            f"{EDA_FINDINGS['duplicates_removed']:,} duplicates removed. "
            f"Class balance: {EDA_FINDINGS['legitimate_pct']:.1f}% legitimate / "
            f"{EDA_FINDINGS['phishing_pct']:.1f}% phishing — no resampling needed. "
            f"No missing values. Top correlated features: {', '.join(EDA_FINDINGS['top_features'])}.",
        "Model Performance":
            f"Six models evaluated:\n{other_str}\n"
            f"Best (tuned {BEST_MODEL_NAME}): Accuracy={BEST_METRICS['Accuracy']:.4f}, "
            f"F1={BEST_METRICS['F1-Score']:.4f}, ROC-AUC={BEST_METRICS['ROC-AUC']:.4f}.",
        "Best Model Justification":
            f"SVM selected for highest tuned F1 ({BEST_METRICS['F1-Score']:.4f}) via GridSearchCV "
            f"(best params: {BEST_METRICS['Best Params']}). Discuss precision/recall tradeoff in security contexts.",
        "Prescriptive Recommendations":
            "Decision logic: Phishing >0.8 conf → block immediately (Critical); "
            "Phishing 0.6–0.8 → warn user (High); Legit >0.8 → allow + safety badge (Low); "
            "Legit 0.6–0.8 → allow with caution (Medium); <0.6 conf → flag for manual review.",
        "Limitations and Future Work":
            "Dataset from 2012; phishing tactics have evolved. Feature encoding loses raw URL nuance. "
            "Suggest: live URL scraping pipeline, deep learning on raw HTML, continual retraining.",
    }

    guidance_blocks = "\n\n".join(
        f"**{s}**: {section_guidance[s]}" for s in active if s in section_guidance
    )

    lang_note = f"\n\nIMPORTANT: Write the entire report in {language}." if language != "English" else ""

    return f"""You are an expert in machine learning and cybersecurity writing a {report_type} for a phishing website detection project.

TONE: {tone_guide.get(tone, tone_guide['Business'])}
FORMAT: {type_guide.get(report_type, type_guide['Detailed Analysis'])}
SECTIONS TO COVER: {sections_str}

SECTION GUIDANCE:
{guidance_blocks}

TOP FEATURES (Random Forest importance): {top_feats}

FORMATTING RULES:
- Use Markdown formatting (##, ### headers, bullet points, bold key numbers)
- No filler phrases like "It is worth noting" or "In conclusion"
- Be direct, specific, and data-driven

Now write the {report_type}:{lang_note}"""

# ─────────────────────────────────────────────────────────────────────────────
# PRESCRIPTIVE RECOMMENDATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def get_recommendation(prediction, confidence):
    if confidence < 0.6:
        return {"priority":"REVIEW",   "icon":"🔍","action":"Flag for Manual Review",
                "message":"Model confidence is low. Manual inspection recommended.",
                "color":"#7a8299","bg":"rgba(122,130,153,0.10)"}
    if prediction == 1:
        if confidence > 0.8:
            return {"priority":"CRITICAL","icon":"🚫","action":"Block Immediately & Alert User",
                    "message":"High-confidence phishing site. Block and notify the user.",
                    "color":"#ff4757","bg":"rgba(255,71,87,0.10)"}
        return     {"priority":"HIGH",   "icon":"⚠️","action":"Display Warning — Require Confirmation",
                    "message":"Likely phishing. Require explicit user confirmation before proceeding.",
                    "color":"#ffa502","bg":"rgba(255,165,2,0.10)"}
    if confidence > 0.8:
        return     {"priority":"LOW",    "icon":"✅","action":"Allow Access — Display Safety Badge",
                    "message":"Site appears legitimate. Display a safety indicator.",
                    "color":"#00ff88","bg":"rgba(0,255,136,0.08)"}
    return         {"priority":"MEDIUM", "icon":"🟡","action":"Allow with Caution — Monitor Behavior",
                    "message":"Appears legitimate with moderate confidence. Monitor activity.",
                    "color":"#ffdd59","bg":"rgba(255,221,89,0.08)"}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem;'>
        <span style='font-family:Space Mono,monospace;font-size:1.2rem;color:#00d4ff;'>🛡️ PHISHGUARD AI</span>
        <p style='color:#7a8299;font-size:0.76rem;margin-top:4px;'>ML-Powered Phishing Detection</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Model status
    if model is not None:
        st.success(f"✅ Model loaded: **{model_info.get('model_name', BEST_MODEL_NAME)}**")
    else:
        st.error("⚠️ Model files not found. Place best_model.joblib, scaler.joblib, and model_info.joblib in the project folder.")

    # Gemini status — no key shown, just connection state
    st.markdown("#### 🤖 Gemini AI")
    if gemini_client:
        st.markdown("<span style='color:#00ff88;font-size:0.82rem;'>● Connected</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#ff4757;font-size:0.82rem;'>● Not connected</span>", unsafe_allow_html=True)
        if gemini_error:
            st.caption(gemini_error)

    st.divider()

    # Best model metrics
    st.markdown("#### 📊 Best Model Metrics")
    for label, val in [
        ("Model",    BEST_MODEL_NAME),
        ("Accuracy", f"{BEST_METRICS['Accuracy']:.2%}"),
        ("F1-Score", f"{BEST_METRICS['F1-Score']:.4f}"),
        ("ROC-AUC",  f"{BEST_METRICS['ROC-AUC']:.4f}"),
    ]:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;padding:6px 10px;
                    background:#131c30;border-radius:8px;margin-bottom:6px;border:1px solid #1e2a42;'>
            <span style='color:#7a8299;font-size:0.82rem;'>{label}</span>
            <span style='color:#00d4ff;font-family:Space Mono,monospace;font-size:0.85rem;'>{val}</span>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='color:#4a5270;font-size:0.75rem;line-height:1.8;'>
        <b style='color:#7a8299;'>Dataset:</b> UCI ID 327<br>
        <b style='color:#7a8299;'>Instances:</b> 11,055 → 5,849<br>
        <b style='color:#7a8299;'>Features:</b> 30 (encoded -1/0/1)<br>
        <b style='color:#7a8299;'>Target:</b> -1 Legit / 1 Phishing
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='padding:1.5rem 0 0.5rem;border-bottom:1px solid #1e2a42;margin-bottom:1.5rem;'>
    <h1 style='font-family:Space Mono,monospace;font-size:1.8rem;margin:0;
               background:linear-gradient(90deg,#00d4ff,#00ff88);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               letter-spacing:0.04em;'>PHISHGUARD AI</h1>
    <p style='color:#7a8299;margin:4px 0 0 2px;font-size:0.9rem;'>
        Phishing Website Detection &nbsp;·&nbsp; UCI Dataset (ID: 327) &nbsp;·&nbsp; Powered by Google Gemini
    </p>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍  Website Analyzer", "📝  AI Report Generator", "📊  Model Dashboard"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — WEBSITE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    if model is None:
        st.error("Model files not found. Place **best_model.joblib**, **scaler.joblib**, and **model_info.joblib** in the same folder as app.py.")
        st.stop()

    st.markdown("##### Select -1 (Legitimate), 0 (Suspicious), or 1 (Phishing) for each website feature.")

    feature_values = {}
    for group_name, group_feats in FEATURE_GROUPS.items():
        with st.expander(group_name, expanded=(group_name == "🌐 URL-Based")):
            cols = st.columns(4)
            for i, feat in enumerate(group_feats):
                with cols[i % 4]:
                    feature_values[feat] = st.selectbox(
                        label=FEATURE_LABELS.get(feat, feat),
                        options=[-1, 0, 1],
                        format_func=lambda x: OPTIONS_MAP[x],
                        key=f"feat_{feat}",
                        index=0,
                    )

    st.markdown("")
    col_btn, _ = st.columns([2, 5])
    with col_btn:
        analyze_btn = st.button("🔍 Analyze Website", type="primary", use_container_width=True)

    if analyze_btn:
        arr = np.array([[feature_values[f] for f in FEATURES]])
        if model_info.get("needs_scaling", False) and scaler is not None:
            arr = scaler.transform(arr)

        prediction = model.predict(arr)[0]
        probs      = model.predict_proba(arr)[0]
        phishing_p = float(probs[1])
        legit_p    = float(probs[0])
        confidence = phishing_p if prediction == 1 else legit_p

        rec        = get_recommendation(prediction, confidence)
        pred_label = "PHISHING DETECTED" if prediction == 1 else "LEGITIMATE WEBSITE"
        pred_icon  = "🚨" if prediction == 1 else "✅"
        pred_color = "#ff4757" if prediction == 1 else "#00ff88"

        st.markdown("---")

        # Main result card
        st.markdown(f"""
        <div style='background:{rec["bg"]};border:1px solid {rec["color"]}40;
                    border-radius:14px;padding:1.5rem 2rem;margin:1rem 0;'>
            <div style='display:flex;align-items:center;gap:1rem;flex-wrap:wrap;'>
                <span style='font-size:2.2rem;'>{pred_icon}</span>
                <div>
                    <div style='font-family:Space Mono,monospace;font-size:1.4rem;
                                color:{pred_color};font-weight:700;letter-spacing:0.05em;'>{pred_label}</div>
                    <div style='color:#7a8299;font-size:0.85rem;margin-top:4px;'>
                        Confidence:
                        <span style='color:{rec["color"]};font-family:Space Mono,monospace;'>{confidence:.2%}</span>
                        &nbsp;·&nbsp; Priority:
                        <span style='color:{rec["color"]};font-weight:600;'>{rec["priority"]}</span>
                    </div>
                </div>
                <div style='margin-left:auto;text-align:right;'>
                    <div style='font-size:0.82rem;color:#7a8299;text-transform:uppercase;letter-spacing:0.06em;'>Recommended Action</div>
                    <div style='color:{rec["color"]};font-weight:600;margin-top:4px;'>{rec["icon"]} {rec["action"]}</div>
                    <div style='color:#7a8299;font-size:0.8rem;margin-top:4px;max-width:320px;'>{rec["message"]}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Probability bars + feature summary
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Probability Breakdown**")
            st.markdown(f"""
            <div style='background:#131c30;border-radius:10px;padding:1rem 1.2rem;border:1px solid #1e2a42;'>
                <div style='margin-bottom:12px;'>
                    <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                        <span style='color:#ff4757;font-size:0.82rem;'>🔴 Phishing</span>
                        <span style='color:#ff4757;font-family:Space Mono,monospace;font-size:0.82rem;'>{phishing_p:.2%}</span>
                    </div>
                    <div style='background:#1e2a42;border-radius:4px;height:8px;overflow:hidden;'>
                        <div style='background:#ff4757;width:{phishing_p*100:.1f}%;height:100%;border-radius:4px;'></div>
                    </div>
                </div>
                <div>
                    <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                        <span style='color:#00ff88;font-size:0.82rem;'>🟢 Legitimate</span>
                        <span style='color:#00ff88;font-family:Space Mono,monospace;font-size:0.82rem;'>{legit_p:.2%}</span>
                    </div>
                    <div style='background:#1e2a42;border-radius:4px;height:8px;overflow:hidden;'>
                        <div style='background:#00ff88;width:{legit_p*100:.1f}%;height:100%;border-radius:4px;'></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        with col_r:
            st.markdown("**Feature Summary**")
            ph_c = sum(1 for v in feature_values.values() if v ==  1)
            su_c = sum(1 for v in feature_values.values() if v ==  0)
            le_c = sum(1 for v in feature_values.values() if v == -1)
            st.markdown(f"""
            <div style='background:#131c30;border-radius:10px;padding:1rem 1.2rem;border:1px solid #1e2a42;'>
                <div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e2a42;'>
                    <span style='color:#7a8299;font-size:0.82rem;'>🔴 Phishing indicators</span>
                    <span style='color:#ff4757;font-family:Space Mono,monospace;font-size:0.82rem;'>{ph_c} / 30</span>
                </div>
                <div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e2a42;'>
                    <span style='color:#7a8299;font-size:0.82rem;'>🟡 Suspicious indicators</span>
                    <span style='color:#ffdd59;font-family:Space Mono,monospace;font-size:0.82rem;'>{su_c} / 30</span>
                </div>
                <div style='display:flex;justify-content:space-between;padding:6px 0;'>
                    <span style='color:#7a8299;font-size:0.82rem;'>🟢 Legitimate indicators</span>
                    <span style='color:#00ff88;font-family:Space Mono,monospace;font-size:0.82rem;'>{le_c} / 30</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with st.expander("📋 Full Feature Input Table"):
            st.dataframe(pd.DataFrame({
                "Feature":        [FEATURE_LABELS.get(f, f) for f in FEATURES],
                "Raw Value":      [feature_values[f] for f in FEATURES],
                "Interpretation": [OPTIONS_MAP[feature_values[f]] for f in FEATURES],
            }), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("##### Generate professional reports from your actual model results using Google Gemini.")

    if gemini_client is None:
        st.error(f"Gemini is unavailable: **{gemini_error}**")
        st.info("Add your API key to `.streamlit/secrets.toml`:\n```toml\nGEMINI_API_KEY = \"AIzaSy...\"\n```")
        st.stop()

    st.divider()

    # Report configuration
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("#### ⚙️ Report Options")
        report_type = st.selectbox("Report Type", [
            "Executive Summary", "Detailed Analysis",
            "Technical Documentation", "Business Presentation",
        ])
        tone     = st.radio("Tone", ["Academic", "Business", "Technical"], horizontal=True)
        language = st.selectbox("Language", ["English", "Spanish", "French", "German", "Filipino"])

    with col_r:
        st.markdown("#### 📋 Sections to Include")
        sections = {
            "Problem Statement":            st.checkbox("Problem Statement",            True,  key="s1"),
            "Dataset Description":          st.checkbox("Dataset Description",          True,  key="s2"),
            "EDA Findings":                 st.checkbox("EDA Findings",                 True,  key="s3"),
            "Model Performance":            st.checkbox("Model Performance",            True,  key="s4"),
            "Best Model Justification":     st.checkbox("Best Model Justification",     True,  key="s5"),
            "Prescriptive Recommendations": st.checkbox("Prescriptive Recommendations", True,  key="s6"),
            "Limitations and Future Work":  st.checkbox("Limitations and Future Work",  False, key="s7"),
        }

    # Prompt editor
    auto_prompt = build_prompt(report_type, tone, sections, language)
    with st.expander("✏️ Edit Prompt (Advanced)"):
        st.markdown("<span style='color:#7a8299;font-size:0.8rem;'>Modify before sending. Leave as-is to use the auto-generated prompt.</span>", unsafe_allow_html=True)
        custom_prompt_text = st.text_area(
            "", value=auto_prompt, height=280, key="prompt_area",
            label_visibility="collapsed",
        )

    final_prompt = custom_prompt_text.strip() if custom_prompt_text.strip() else auto_prompt
    if language != "English" and "IMPORTANT: Write the entire report in" not in final_prompt:
        final_prompt += f"\n\nIMPORTANT: Write the entire report in {language}."

    st.markdown("")
    g_col, _ = st.columns([2, 5])
    with g_col:
        gen_btn = st.button("🚀 Generate Report", type="primary", use_container_width=True)

    if gen_btn:
        with st.spinner("✨ Gemini is writing your report..."):
            report_out = generate_report(final_prompt)
        st.session_state.latest_report = report_out
        st.session_state.report_history.append({
            "type":      report_type,
            "tone":      tone,
            "language":  language,
            "content":   report_out,
            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
        })

    # Display latest report
    if st.session_state.latest_report:
        report_text = st.session_state.latest_report
        st.divider()
        st.markdown("### Generated Report")
        st.markdown("""
        <div style='background:#0f1628;border:1px solid #1e2a42;border-radius:14px;
                    padding:2rem;margin-bottom:1rem;line-height:1.85;color:#e8eaf0;'>
        """, unsafe_allow_html=True)
        st.markdown(report_text)
        st.markdown("</div>", unsafe_allow_html=True)

        # Export + actions row
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            st.download_button(
                "📥 Download .md", report_text,
                file_name=f"phishing_report_{report_type.lower().replace(' ','_')}.md",
                mime="text/markdown",
            )
        with ec2:
            st.download_button(
                "📄 Download .txt", report_text,
                file_name=f"phishing_report_{report_type.lower().replace(' ','_')}.txt",
                mime="text/plain",
            )
        with ec3:
            if st.button("🔄 Regenerate", key="regen_btn"):
                st.session_state.latest_report = None
                st.rerun()
        with ec4:
            feedback = st.selectbox(
                "Refine:", ["—", "Make it more technical", "Simplify language",
                            "Make it shorter", "Add more data details"],
                key="feedback_sel", label_visibility="collapsed",
            )
            if feedback != "—" and st.button("Apply", key="apply_fb"):
                with st.spinner("Refining..."):
                    refined = generate_report(final_prompt + f"\n\nAdditional instruction: {feedback}.")
                st.session_state.latest_report = refined
                st.rerun()

    # Report history
    if st.session_state.report_history:
        with st.expander(f"🕑 Report History ({len(st.session_state.report_history)} generated)"):
            for i, rep in enumerate(reversed(st.session_state.report_history[-6:])):
                hc1, hc2 = st.columns([5, 1])
                with hc1:
                    st.markdown(
                        f"<span style='color:#00d4ff;font-family:Space Mono,monospace;font-size:0.82rem;'>{rep['timestamp']}</span>"
                        f"<span style='color:#7a8299;font-size:0.82rem;margin-left:10px;'>{rep['type']} · {rep['tone']} · {rep['language']}</span>",
                        unsafe_allow_html=True,
                    )
                with hc2:
                    if st.button("Load", key=f"load_{i}"):
                        st.session_state.latest_report = rep["content"]
                        st.rerun()
                st.markdown("<hr style='border-color:#1e2a42;margin:6px 0;'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("#### All Model Results — Base vs. Tuned")

    rows = []
    for name, m in MODEL_RESULTS.items():
        tuned = TUNED_RESULTS.get(name, {})
        rows.append({
            "Model":         name,
            "Base Accuracy": f"{m['Accuracy']:.4f}",
            "Base F1":       f"{m['F1-Score']:.4f}",
            "Base ROC-AUC":  f"{m['ROC-AUC']:.4f}",
            "Tuned F1":      f"{tuned['F1-Score']:.4f}" if tuned else "—",
            "Tuned ROC-AUC": f"{tuned['ROC-AUC']:.4f}" if tuned else "—",
            "Best Params":   tuned.get("Best Params", "—") if tuned else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(f"#### 🏆 Best Model: {BEST_MODEL_NAME} (tuned)")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Accuracy",    f"{BEST_METRICS['Accuracy']:.2%}")
    with m2: st.metric("F1-Score",    f"{BEST_METRICS['F1-Score']:.4f}")
    with m3: st.metric("ROC-AUC",     f"{BEST_METRICS['ROC-AUC']:.4f}")
    with m4: st.metric("Best Params", BEST_METRICS["Best Params"])

    st.divider()
    st.markdown("#### 🌲 Top 10 Features — Random Forest Importance")
    fi_df    = pd.DataFrame(FEATURE_IMPORTANCE.items(), columns=["Feature","Importance"]).sort_values("Importance", ascending=False)
    max_imp  = fi_df["Importance"].max()
    for _, row in fi_df.iterrows():
        bar_color = "#00d4ff" if row["Importance"] > 0.07 else "#2a3a5c"
        pct       = row["Importance"] / max_imp * 100
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:12px;margin-bottom:8px;'>
            <div style='width:190px;color:#7a8299;font-size:0.82rem;text-align:right;flex-shrink:0;'>{row["Feature"]}</div>
            <div style='flex:1;background:#1e2a42;border-radius:4px;height:10px;overflow:hidden;'>
                <div style='background:{bar_color};width:{pct:.1f}%;height:100%;border-radius:4px;'></div>
            </div>
            <div style='width:52px;color:#00d4ff;font-family:Space Mono,monospace;font-size:0.8rem;'>{row["Importance"]:.3f}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📋 EDA Summary")
    ea, eb = st.columns(2)
    with ea:
        st.markdown(f"""
        <div style='background:#131c30;border-radius:12px;padding:1.2rem 1.5rem;border:1px solid #1e2a42;'>
            <p style='color:#7a8299;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.06em;margin:0 0 10px;'>Dataset Stats</p>
            <div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1e2a42;'>
                <span style='color:#7a8299;font-size:0.83rem;'>Total instances</span>
                <span style='color:#00d4ff;font-family:Space Mono,monospace;'>{EDA_FINDINGS['total_instances']:,}</span>
            </div>
            <div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1e2a42;'>
                <span style='color:#7a8299;font-size:0.83rem;'>After deduplication</span>
                <span style='color:#00d4ff;font-family:Space Mono,monospace;'>{EDA_FINDINGS['after_dedup']:,}</span>
            </div>
            <div style='display:flex;justify-content:space-between;padding:5px 0;'>
                <span style='color:#7a8299;font-size:0.83rem;'>Duplicates removed</span>
                <span style='color:#ffa502;font-family:Space Mono,monospace;'>{EDA_FINDINGS['duplicates_removed']:,}</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with eb:
        st.markdown(f"""
        <div style='background:#131c30;border-radius:12px;padding:1.2rem 1.5rem;border:1px solid #1e2a42;'>
            <p style='color:#7a8299;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.06em;margin:0 0 10px;'>Class Balance</p>
            <div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1e2a42;'>
                <span style='color:#7a8299;font-size:0.83rem;'>🟢 Legitimate (-1)</span>
                <span style='color:#00ff88;font-family:Space Mono,monospace;'>{EDA_FINDINGS['legitimate_pct']:.1f}%</span>
            </div>
            <div style='display:flex;justify-content:space-between;padding:5px 0;'>
                <span style='color:#7a8299;font-size:0.83rem;'>🔴 Phishing (1)</span>
                <span style='color:#ff4757;font-family:Space Mono,monospace;'>{EDA_FINDINGS['phishing_pct']:.1f}%</span>
            </div>
            <p style='color:#4a5270;font-size:0.78rem;margin:10px 0 0;'>Near-perfect balance — no resampling needed.</p>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='text-align:center;color:#4a5270;font-size:0.75rem;
            padding:2rem 0 1rem;border-top:1px solid #1e2a42;margin-top:2rem;
            font-family:Space Mono,monospace;'>
    PHISHGUARD AI &nbsp;·&nbsp; UCI Phishing Websites (ID: 327) &nbsp;·&nbsp; scikit-learn + Google Gemini
</div>""", unsafe_allow_html=True)
