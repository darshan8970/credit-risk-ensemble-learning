"""CreditWise AI - Clean & Modern Credit Risk Intelligence Dashboard.

Machine Learning Ensemble Platform for German Credit Risk Assessment.
"""

import io
import json
import os
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import ASSETS_DIR, CATEGORY_MAPPINGS, MODELS_DIR, REPORTS_DIR
from src.predict import (
    interpret_credit_risk,
    list_saved_models,
    load_model,
    predict_credit_risk,
)

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditWise AI | Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAMES = {
    "random_forest_model": "Random Forest",
    "bagging_model": "Bagging Classifier",
    "adaboost_model": "AdaBoost",
}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #f8fafc; color: #0f172a; }
.block-container { max-width: 1300px; padding: 1.8rem 2rem 3rem; }
header { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

/* Clear, High-Contrast Labels for All Dropdowns and Widgets */
label, [data-testid="stWidgetLabel"] p {
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin-bottom: 4px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #0b1329 !important; border-right: 1px solid #1e293b; }
section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }

/* Hero Banner */
.hero {
    background: linear-gradient(135deg, #0b1329 0%, #1e293b 50%, #1e3a8a 100%);
    border-radius: 16px; padding: 26px 30px; color: white; margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
}
.hero h1 { font-size: 1.85rem; font-weight: 800; margin: 4px 0 6px; color: #fff; }
.hero p { color: #cbd5e1; font-size: 0.95rem; margin: 0; max-width: 750px; }

/* Cards & Badges */
.card { background: white; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; box-shadow: 0 3px 12px rgba(0,0,0,0.03); margin-bottom: 16px; }
.score-box { border-radius: 14px; padding: 20px; text-align: center; border: 1px solid; margin-bottom: 14px; }
.score-safe { background: #f0fdf4; border-color: #86efac; color: #166534; }
.score-warn { background: #fffbeb; border-color: #fde68a; color: #92400e; }
.score-danger { background: #fef2f2; border-color: #fca5a5; color: #991b1b; }
.score-num { font-family: 'JetBrains Mono', monospace; font-size: 2.5rem; font-weight: 800; margin: 4px 0; }
.model-row { background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
.badge-safe { background: #dcfce7; color: #15803d; padding: 4px 10px; border-radius: 20px; font-weight: 700; font-size: 0.78rem; }
.badge-danger { background: #fee2e2; color: #b91c1c; padding: 4px 10px; border-radius: 20px; font-weight: 700; font-size: 0.78rem; }

/* Crisp High-Contrast Buttons */
.stButton > button {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1rem !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #eff6ff !important;
    color: #1d4ed8 !important;
    border-color: #3b82f6 !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2) !important;
    transform: translateY(-1px);
}
.stButton > button p, .stButton > button span, .stButton > button div {
    color: #0f172a !important;
    font-weight: 700 !important;
}
.stButton > button:hover p, .stButton > button:hover span, .stButton > button:hover div {
    color: #1d4ed8 !important;
}

.stDownloadButton > button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.2rem !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
}
.stDownloadButton > button p, .stDownloadButton > button span {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #f1f5f9;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 700;
    font-size: 0.92rem;
}
.stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] span {
    color: #475569 !important;
    font-weight: 700 !important;
}
.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}
.stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {
    color: #1d4ed8 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. MODEL & ASSET LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    models = {name: load_model(name) for name in list_saved_models()}
    prep_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    if not models or not os.path.exists(prep_path):
        st.error("Models or Preprocessor missing! Run `python main.py` first.")
        st.stop()
    return models, joblib.load(prep_path)

MODELS, PREPROCESSOR = load_resources()

def get_report_json(filename: str):
    path = os.path.join(REPORTS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# -----------------------------------------------------------------------------
# 3. PRESETS & PREDICTION LOGIC
# -----------------------------------------------------------------------------
PRESETS = {
    "🌟 Prime Borrower (Low Risk)": {
        "checking_status": "A13", "duration": 12, "credit_history": "A34", "purpose": "A41",
        "credit_amount": 2500, "savings_status": "A64", "employment_since": "A75",
        "installment_rate": 1, "personal_status_sex": "A93", "other_debtors": "A101",
        "residence_since": 4, "property": "A121", "age": 42, "other_installment_plans": "A143",
        "housing": "A152", "existing_credits": 1, "job": "A174", "num_dependents": 1,
        "telephone": "A192", "foreign_worker": "A201"
    },
    "⚠️ Subprime Applicant (High Risk)": {
        "checking_status": "A11", "duration": 48, "credit_history": "A31", "purpose": "A49",
        "credit_amount": 9500, "savings_status": "A61", "employment_since": "A71",
        "installment_rate": 4, "personal_status_sex": "A91", "other_debtors": "A101",
        "residence_since": 1, "property": "A124", "age": 22, "other_installment_plans": "A141",
        "housing": "A151", "existing_credits": 3, "job": "A171", "num_dependents": 2,
        "telephone": "A191", "foreign_worker": "A201"
    },
    "💼 Salaried Professional": {
        "checking_status": "A12", "duration": 24, "credit_history": "A32", "purpose": "A40",
        "credit_amount": 4200, "savings_status": "A62", "employment_since": "A73",
        "installment_rate": 2, "personal_status_sex": "A93", "other_debtors": "A101",
        "residence_since": 2, "property": "A122", "age": 30, "other_installment_plans": "A143",
        "housing": "A151", "existing_credits": 1, "job": "A173", "num_dependents": 1,
        "telephone": "A192", "foreign_worker": "A201"
    }
}

if "applicant" not in st.session_state:
    st.session_state["applicant"] = PRESETS["💼 Salaried Professional"].copy()

def score_applicant(data: Dict[str, Any]):
    df = pd.DataFrame([data])
    processed = PREPROCESSOR.transform(df)
    results, probs = {}, []
    for name, model in MODELS.items():
        pred = interpret_credit_risk(predict_credit_risk(model, processed))
        prob = float(model.predict_proba(processed)[0][1]) if hasattr(model, "predict_proba") else (1.0 if pred == "Default" else 0.0)
        results[name] = {"pred": pred, "prob": prob}
        probs.append(prob)
    
    avg_p = float(np.mean(probs))
    score = int(np.clip(950 - (avg_p * 630), 300, 950))
    tier = "Low Risk (Approved)" if avg_p < 0.32 else ("Moderate Risk (Review)" if avg_p < 0.50 else "High Default Risk")
    tier_css = "safe" if avg_p < 0.32 else ("warn" if avg_p < 0.50 else "danger")
    return results, avg_p, score, tier, tier_css

# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ CreditWise **AI**")
    st.caption("Credit Risk ML Intelligence Platform")
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "🎯 Credit Risk Assessment",
            "📊 Model Diagnostics",
            "ℹ️ System Info",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("🟢 **System Active**")
    st.caption(f"Ensemble Models: **{len(MODELS)}** | Features: **20**")

# -----------------------------------------------------------------------------
# 5. PAGE: 🎯 CREDIT RISK ASSESSMENT
# -----------------------------------------------------------------------------
if page == "🎯 Credit Risk Assessment":
    st.markdown("""
    <div class="hero">
        <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#93c5fd;">Interactive Underwriting Engine</div>
        <h1>Credit Risk Assessment Workspace</h1>
        <p>Input applicant credit attributes below or load benchmark personas to evaluate loan risk across the trained ensemble.</p>
    </div>
    """, unsafe_allow_html=True)

    # Preset selector
    st.markdown("**🚀 Quick-Load Benchmark Personas:**")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🌟 Prime Borrower (Low Risk)", use_container_width=True):
            st.session_state["applicant"] = PRESETS["🌟 Prime Borrower (Low Risk)"].copy()
            st.rerun()
    with c2:
        if st.button("⚠️ Subprime Applicant (High Risk)", use_container_width=True):
            st.session_state["applicant"] = PRESETS["⚠️ Subprime Applicant (High Risk)"].copy()
            st.rerun()
    with c3:
        if st.button("💼 Salaried Professional (Moderate)", use_container_width=True):
            st.session_state["applicant"] = PRESETS["💼 Salaried Professional"].copy()
            st.rerun()

    st.write("")
    col_in, col_out = st.columns([1.15, 0.85], gap="large")

    with col_in:
        t1, t2, t3 = st.tabs(["🏛️ Financial Accounts", "💰 Loan Request Details", "👤 Applicant Profile"])
        
        with t1:
            st.markdown("##### 1. Financial Standing & Stability")
            f1, f2 = st.columns(2)
            with f1:
                st.session_state["applicant"]["checking_status"] = st.selectbox(
                    "Checking Account Balance Status",
                    options=list(CATEGORY_MAPPINGS["checking_status"].keys()),
                    format_func=CATEGORY_MAPPINGS["checking_status"].get,
                    index=list(CATEGORY_MAPPINGS["checking_status"].keys()).index(st.session_state["applicant"]["checking_status"]),
                    help="Balance status of primary bank checking account",
                )
                st.session_state["applicant"]["credit_history"] = st.selectbox(
                    "Credit Repayment History Track Record",
                    options=list(CATEGORY_MAPPINGS["credit_history"].keys()),
                    format_func=CATEGORY_MAPPINGS["credit_history"].get,
                    index=list(CATEGORY_MAPPINGS["credit_history"].keys()).index(st.session_state["applicant"]["credit_history"]),
                    help="Historical loan repayment track record",
                )
            with f2:
                st.session_state["applicant"]["savings_status"] = st.selectbox(
                    "Savings & Deposit Balance Status",
                    options=list(CATEGORY_MAPPINGS["savings_status"].keys()),
                    format_func=CATEGORY_MAPPINGS["savings_status"].get,
                    index=list(CATEGORY_MAPPINGS["savings_status"].keys()).index(st.session_state["applicant"]["savings_status"]),
                    help="Liquid savings buffer available with bank",
                )
                st.session_state["applicant"]["employment_since"] = st.selectbox(
                    "Employment Duration with Current Employer",
                    options=list(CATEGORY_MAPPINGS["employment_since"].keys()),
                    format_func=CATEGORY_MAPPINGS["employment_since"].get,
                    index=list(CATEGORY_MAPPINGS["employment_since"].keys()).index(st.session_state["applicant"]["employment_since"]),
                    help="Length of continuous employment",
                )

        with t2:
            st.markdown("##### 2. Loan Parameters & Terms")
            l1, l2 = st.columns(2)
            with l1:
                st.session_state["applicant"]["credit_amount"] = st.number_input(
                    "Requested Loan Amount (INR ₹)",
                    min_value=250, max_value=100000, value=int(st.session_state["applicant"]["credit_amount"]), step=250,
                    help="Total credit principal applied for",
                )
                st.session_state["applicant"]["duration"] = st.slider(
                    "Repayment Duration (Months)",
                    min_value=4, max_value=72, value=int(st.session_state["applicant"]["duration"]), step=2,
                    help="Duration of the loan in months",
                )
                st.session_state["applicant"]["installment_rate"] = st.select_slider(
                    "Installment Rate Tier (% of Monthly Income)",
                    options=[1, 2, 3, 4], value=int(st.session_state["applicant"]["installment_rate"]),
                    format_func=lambda x: f"Tier {x} ({x*10}% of monthly income)",
                )
            with l2:
                st.session_state["applicant"]["purpose"] = st.selectbox(
                    "Loan Financing Purpose / Objective",
                    options=list(CATEGORY_MAPPINGS["purpose"].keys()),
                    format_func=CATEGORY_MAPPINGS["purpose"].get,
                    index=list(CATEGORY_MAPPINGS["purpose"].keys()).index(st.session_state["applicant"]["purpose"]),
                )
                st.session_state["applicant"]["existing_credits"] = st.number_input(
                    "Number of Existing Loans at this Bank",
                    min_value=1, max_value=10, value=int(st.session_state["applicant"]["existing_credits"]),
                )
                st.session_state["applicant"]["other_installment_plans"] = st.selectbox(
                    "Other Active External EMI / Installment Plans",
                    options=list(CATEGORY_MAPPINGS["other_installment_plans"].keys()),
                    format_func=CATEGORY_MAPPINGS["other_installment_plans"].get,
                    index=list(CATEGORY_MAPPINGS["other_installment_plans"].keys()).index(st.session_state["applicant"]["other_installment_plans"]),
                )

        with t3:
            st.markdown("##### 3. Demographics & Collateral")
            p1, p2 = st.columns(2)
            with p1:
                st.session_state["applicant"]["age"] = st.slider("Applicant Age (Years)", 18, 80, int(st.session_state["applicant"]["age"]))
                st.session_state["applicant"]["housing"] = st.selectbox(
                    "Current Housing & Residence Status",
                    options=list(CATEGORY_MAPPINGS["housing"].keys()),
                    format_func=CATEGORY_MAPPINGS["housing"].get,
                    index=list(CATEGORY_MAPPINGS["housing"].keys()).index(st.session_state["applicant"]["housing"]),
                )
                st.session_state["applicant"]["property"] = st.selectbox(
                    "Property & Collateral Assets Owned",
                    options=list(CATEGORY_MAPPINGS["property"].keys()),
                    format_func=CATEGORY_MAPPINGS["property"].get,
                    index=list(CATEGORY_MAPPINGS["property"].keys()).index(st.session_state["applicant"]["property"]),
                )
                st.session_state["applicant"]["residence_since"] = st.select_slider("Years at Current Address", [1, 2, 3, 4], int(st.session_state["applicant"]["residence_since"]))
            with p2:
                st.session_state["applicant"]["job"] = st.selectbox(
                    "Employment / Profession Type",
                    options=list(CATEGORY_MAPPINGS["job"].keys()),
                    format_func=CATEGORY_MAPPINGS["job"].get,
                    index=list(CATEGORY_MAPPINGS["job"].keys()).index(st.session_state["applicant"]["job"]),
                )
                st.session_state["applicant"]["other_debtors"] = st.selectbox(
                    "Co-Applicant / Guarantor Support",
                    options=list(CATEGORY_MAPPINGS["other_debtors"].keys()),
                    format_func=CATEGORY_MAPPINGS["other_debtors"].get,
                    index=list(CATEGORY_MAPPINGS["other_debtors"].keys()).index(st.session_state["applicant"]["other_debtors"]),
                )
                st.session_state["applicant"]["personal_status_sex"] = st.selectbox(
                    "Personal Status & Gender",
                    options=list(CATEGORY_MAPPINGS["personal_status_sex"].keys()),
                    format_func=CATEGORY_MAPPINGS["personal_status_sex"].get,
                    index=list(CATEGORY_MAPPINGS["personal_status_sex"].keys()).index(st.session_state["applicant"]["personal_status_sex"]),
                )
                st.session_state["applicant"]["foreign_worker"] = st.selectbox(
                    "Foreign Worker Status",
                    options=list(CATEGORY_MAPPINGS["foreign_worker"].keys()),
                    format_func=CATEGORY_MAPPINGS["foreign_worker"].get,
                    index=list(CATEGORY_MAPPINGS["foreign_worker"].keys()).index(st.session_state["applicant"]["foreign_worker"]),
                )

    # Perform inference & display decision intelligence
    results, avg_p, score, tier, tier_css = score_applicant(st.session_state["applicant"])

    with col_out:
        st.markdown("##### 📊 Decision Intelligence")
        st.markdown(f"""
        <div class="score-box score-{tier_css}">
            <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;">Credit Health Index</div>
            <div class="score-num">{score} <span style="font-size:1rem; color:#64748b;">/ 1000</span></div>
            <div style="font-weight:800; font-size:1.1rem;">{tier}</div>
            <div style="font-size:0.85rem; margin-top:6px; color:#475569;">Consensus Default Probability: <b>{avg_p:.1%}</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Ensemble Models Breakdown:**")
        for m_key, r in results.items():
            name = MODEL_NAMES.get(m_key, m_key)
            is_def = r["pred"] == "Default"
            badge = "badge-danger" if is_def else "badge-safe"
            label = "Default Risk" if is_def else "Low Risk / Safe"
            st.markdown(f"""
            <div class="model-row">
                <div>
                    <div style="font-weight:700; font-size:0.92rem;">{name}</div>
                    <div style="font-size:0.78rem; color:#64748b;">Default Likelihood: <b>{r['prob']:.1%}</b></div>
                </div>
                <span class="{badge}">{label}</span>
            </div>
            """, unsafe_allow_html=True)

        # Action Recommendation
        st.markdown(f"""
        <div class="card" style="margin-top:14px; border-left:4px solid #2563eb;">
            <div style="font-weight:700; font-size:0.9rem; color:#2563eb; margin-bottom:4px;">📌 Underwriting Policy Suggestion:</div>
            <div style="font-size:0.85rem; color:#475569; line-height:1.5;">
                {"✅ Standard approval recommended under standard prime interest rates." if avg_p < 0.32 else ("⚠️ Conditional approval recommended. Request secondary verification or down-payment." if avg_p < 0.50 else "❌ High risk of loan default. Collateral or verified guarantor mandatory before loan sanction.")}
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. PAGE: 📊 MODEL DIAGNOSTICS
# -----------------------------------------------------------------------------
elif page == "📊 Model Diagnostics":
    st.markdown("""
    <div class="hero">
        <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#93c5fd;">Evaluation Hub</div>
        <h1>Model Evaluation & Diagnostics</h1>
        <p>Inspect performance metrics, ROC/PR curves, confusion matrices, and hypothesis tests across trained ensemble algorithms.</p>
    </div>
    """, unsafe_allow_html=True)

    metrics = get_report_json("metrics_summary.json")
    if metrics:
        st.markdown("##### 🏆 Model Performance Leaderboard")
        m_cols = st.columns(len(metrics.get("metrics", metrics)))
        for col, (m_name, vals) in zip(m_cols, metrics.get("metrics", metrics).items()):
            if isinstance(vals, dict):
                with col:
                    st.markdown(f"""
                    <div class="card" style="border-top:3px solid #2563eb;">
                        <div style="font-size:0.78rem; font-weight:700; color:#64748b; text-transform:uppercase;">{m_name}</div>
                        <div style="font-size:1.4rem; font-weight:800; color:#0f172a; margin:4px 0 10px;">ROC-AUC: {vals.get('ROC-AUC', 'N/A')}</div>
                        <div style="font-size:0.85rem; color:#475569;">Accuracy: <b>{vals.get('Accuracy', 0):.1%}</b></div>
                        <div style="font-size:0.85rem; color:#475569;">F1-Score: <b>{vals.get('F1-Score', 'N/A')}</b></div>
                        <div style="font-size:0.85rem; color:#475569;">Precision: <b>{vals.get('Precision', 0):.1%}</b></div>
                    </div>
                    """, unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["📈 ROC Curves", "🎯 PR Curves", "🔲 Confusion Matrices", "💡 Feature Importance", "🔬 Hypothesis Tests"])
    tab_files = [(t1, "roc_curves.png"), (t2, "precision_recall_curves.png"), (t3, "confusion_matrices.png"), (t4, "feature_importance.png")]
    for tab, img_name in tab_files:
        with tab:
            img_path = os.path.join(ASSETS_DIR, img_name)
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.info("Chart asset not found.")

    with t5:
        hypo = get_report_json("hypothesis_test_results.json")
        if hypo:
            st.markdown(f"**Conclusion:** {hypo.get('Conclusion', {}).get('Summary', '')}")
            st.json(hypo)

# -----------------------------------------------------------------------------
# 8. PAGE: ℹ️ SYSTEM INFO
# -----------------------------------------------------------------------------
else:
    st.markdown("""
    <div class="hero">
        <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#93c5fd;">System Overview</div>
        <h1>Credit Risk Ensemble Architecture</h1>
        <p>A production-ready pipeline utilizing Random Forest, Bagging, and AdaBoost trained on the German Credit dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><b>🌲 Random Forest</b><br><small>De-correlated decision tree ensemble with randomized feature selection for high generalization.</small></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><b>📦 Bagging</b><br><small>Bootstrap aggregation over multiple base decision estimators to reduce prediction variance.</small></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><b>🚀 AdaBoost</b><br><small>Adaptive iterative boosting focusing learning weights on difficult border cases.</small></div>', unsafe_allow_html=True)
