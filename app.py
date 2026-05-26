import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }
section[data-testid="stSidebar"] { background: #0f1117; color: #fff; }
section[data-testid="stSidebar"] label { color: #ccc !important; font-size: 0.82rem; letter-spacing: 0.05em; text-transform: uppercase; }
.result-approved { background: linear-gradient(135deg, #00c9a7, #00a67c); color: white; border-radius: 16px; padding: 2rem; text-align: center; font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; box-shadow: 0 8px 32px rgba(0,201,167,0.3); }
.result-rejected { background: linear-gradient(135deg, #ff6b6b, #c0392b); color: white; border-radius: 16px; padding: 2rem; text-align: center; font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; box-shadow: 0 8px 32px rgba(255,107,107,0.3); }
.metric-box { background: white; border-radius: 12px; padding: 1.2rem 1.5rem; border-left: 4px solid #5c6bc0; margin-bottom: 0.8rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
.metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-value { font-size: 1.3rem; font-weight: 700; color: #1a1a2e; font-family: 'Syne', sans-serif; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model  = joblib.load("loan_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

st.markdown("# 🏦 Loan Approval Predictor")
st.markdown("##### Powered by Gaussian Naive Bayes · 89.5% accuracy")
st.divider()

if not model_loaded:
    st.error("⚠️ `loan_model.pkl` or `scaler.pkl` not found. Place them in the same folder as `app.py`.")
    st.stop()

# ── Exact 12 features from scaler.feature_names_in_ ──────────────────────────
# Applicant_Income, Coapplicant_Income, Age, Dependents, Credit_Score,
# Existing_Loans, DTI_Ratio, Savings, Collateral_Value, Loan_Amount,
# Loan_Term, Education_Level

with st.sidebar:
    st.markdown("## 📋 Applicant Details")

    applicant_income   = st.number_input("Applicant Income (₹)", 2009, 19988, 10000, step=500)
    coapplicant_income = st.number_input("Co-applicant Income (₹)", 0, 9996, 3000, step=500)
    age                = st.slider("Age", 21, 59, 35)
    dependents         = st.selectbox("Dependents", [0, 1, 2, 3])
    credit_score       = st.slider("Credit Score", 550, 799, 680)
    existing_loans     = st.selectbox("Existing Loans", [0, 1, 2, 3, 4])
    dti_ratio          = st.slider("DTI Ratio", 0.10, 0.60, 0.30, step=0.01,
                                   help="Debt-to-Income ratio (0.10 = 10%)")
    savings            = st.number_input("Savings (₹)", 65, 19996, 8000, step=500)
    collateral_value   = st.number_input("Collateral Value (₹)", 36, 49954, 20000, step=1000)
    loan_amount        = st.number_input("Loan Amount (₹)", 1015, 39995, 15000, step=1000)
    loan_term          = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84])
    education_level    = st.radio("Education Level", ["Graduate (1)", "Not Graduate (0)"])

    predict_btn = st.button("🔍 Predict Now", use_container_width=True, type="primary")

edu = 1 if "Graduate (1)" in education_level else 0

# Exact feature order matching scaler
input_features = np.array([[
    applicant_income,
    coapplicant_income,
    age,
    dependents,
    credit_score,
    existing_loans,
    dti_ratio,
    savings,
    collateral_value,
    loan_amount,
    loan_term,
    edu,
]])

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📊 Input Summary")
    summary = {
        "Applicant Income": f"₹{applicant_income:,}",
        "Co-applicant Income": f"₹{coapplicant_income:,}",
        "Age": f"{age} yrs",
        "Dependents": dependents,
        "Credit Score": credit_score,
        "Existing Loans": existing_loans,
        "DTI Ratio": f"{dti_ratio:.0%}",
        "Savings": f"₹{savings:,}",
        "Collateral Value": f"₹{collateral_value:,}",
        "Loan Amount": f"₹{loan_amount:,}",
        "Loan Term": f"{loan_term} months",
        "Education Level": "Graduate" if edu == 1 else "Not Graduate",
    }
    for label, value in summary.items():
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("### 🎯 Prediction Result")

    if predict_btn:
        scaled     = scaler.transform(input_features)
        prediction = model.predict(scaled)[0]
        proba      = model.predict_proba(scaled)[0]

        if prediction == 1:
            st.markdown(f'<div class="result-approved">✅ APPROVED<br><span style="font-size:1rem;font-weight:400">Confidence: {proba[1]*100:.1f}%</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-rejected">❌ REJECTED<br><span style="font-size:1rem;font-weight:400">Confidence: {proba[0]*100:.1f}%</span></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Approval Probability**")
        st.progress(int(proba[1] * 100))
        st.caption(f"Approved: {proba[1]*100:.1f}%  |  Rejected: {proba[0]*100:.1f}%")
    else:
        st.info("👈 Fill in the details and click **Predict Now**")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📈 Model Performance")
        for k, v in {"Accuracy":"89.5%","Precision":"93.2%","Recall":"69.5%","F1 Score":"79.6%"}.items():
            st.markdown(f'<div class="metric-box"><div class="metric-label">{k}</div><div class="metric-value">{v}</div></div>', unsafe_allow_html=True)

st.divider()
st.caption("Model: GaussianNB · 12 features: Applicant_Income, Coapplicant_Income, Age, Dependents, Credit_Score, Existing_Loans, DTI_Ratio, Savings, Collateral_Value, Loan_Amount, Loan_Term, Education_Level")