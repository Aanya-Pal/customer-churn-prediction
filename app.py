import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_model()
model    = bundle["model"]
le_ff    = bundle["le_ff"]
le_acc   = bundle["le_acc"]
le_bh    = bundle["le_bh"]
le_inc   = bundle["le_inc"]
acc      = bundle["accuracy"]

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F4F8FB; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #028090; }
    .metric-label { font-size: 0.85rem; color: #64748B; margin-top: 4px; }
    .predict-btn > button {
        background-color: #028090 !important;
        color: white !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        padding: 0.6rem 2rem !important;
    }
    .result-churn {
        background: #FEF3C7; border-left: 5px solid #F59E0B;
        padding: 1rem 1.5rem; border-radius: 8px; margin-top: 1rem;
    }
    .result-safe {
        background: #D1FAE5; border-left: 5px solid #028090;
        padding: 1rem 1.5rem; border-radius: 8px; margin-top: 1rem;
    }
    h1 { color: #0D1B2A !important; }
    h2, h3 { color: #0D1B2A !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 📊 Customer Churn Predictor")
st.markdown("**Random Forest Classifier** | B.Tech Gen AI — Final Project")
st.markdown("---")

# ── Top metrics ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{acc*100:.1f}%</div>
        <div class="metric-label">Model Accuracy</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="metric-card">
        <div class="metric-value">100</div>
        <div class="metric-label">Trees in Forest</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="metric-card">
        <div class="metric-value">6</div>
        <div class="metric-label">Input Features</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""<div class="metric-card">
        <div class="metric-value">2</div>
        <div class="metric-label">Output Classes</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Layout: Input | Results ────────────────────────────────────────────────────
left, right = st.columns([1, 1.4], gap="large")

with left:
    st.subheader("🔍 Customer Details")
    st.markdown("Fill in the customer attributes below:")

    age = st.slider("Age", min_value=18, max_value=70, value=30, step=1)
    frequent_flyer = st.selectbox("Frequent Flyer", options=["No", "Yes"])
    income_class = st.selectbox("Annual Income Class", options=["Low Income", "Middle Income", "High Income"])
    services = st.slider("Number of Services Opted", min_value=1, max_value=6, value=3, step=1)
    account_synced = st.selectbox("Account Synced to Social Media", options=["No", "Yes"])
    booked_hotel = st.selectbox("Booked Hotel", options=["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Predict Churn", use_container_width=True)

with right:
    st.subheader("📈 Prediction Result")

    if predict_clicked:
        # Encode inputs
        ff_enc  = le_ff.transform([frequent_flyer])[0]
        acc_enc = le_acc.transform([account_synced])[0]
        bh_enc  = le_bh.transform([booked_hotel])[0]
        inc_enc = le_inc.transform([income_class])[0]

        X_input = pd.DataFrame([[age, ff_enc, inc_enc, services, acc_enc, bh_enc]],
                               columns=["Age","FrequentFlyer_enc","AnnualIncomeClass_enc",
                                        "ServicesOpted","AccountSyncedToSocialMedia_enc",
                                        "BookedHotelOrNot_enc"])

        pred        = model.predict(X_input)[0]
        proba       = model.predict_proba(X_input)[0]
        churn_prob  = proba[1] * 100
        safe_prob   = proba[0] * 100

        if pred == 1:
            st.markdown(f"""<div class="result-churn">
                <h3 style="color:#B45309; margin:0">⚠️ High Churn Risk</h3>
                <p style="margin:0.4rem 0 0 0; color:#92400E">
                This customer is likely to churn.<br>
                <strong>Churn Probability: {churn_prob:.1f}%</strong>
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-safe">
                <h3 style="color:#065F46; margin:0">✅ Low Churn Risk</h3>
                <p style="margin:0.4rem 0 0 0; color:#064E3B">
                This customer is likely to stay.<br>
                <strong>Retention Probability: {safe_prob:.1f}%</strong>
                </p>
            </div>""", unsafe_allow_html=True)

        # Probability bar chart
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 1.8))
        fig.patch.set_facecolor("#F4F8FB")
        ax.set_facecolor("#F4F8FB")

        bars = ax.barh(["Will Stay", "Will Churn"],
                       [safe_prob, churn_prob],
                       color=["#028090", "#F59E0B"],
                       height=0.5, edgecolor="none")
        for bar, val in zip(bars, [safe_prob, churn_prob]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=11, color="#0D1B2A", fontweight="bold")

        ax.set_xlim(0, 115)
        ax.set_xlabel("Probability (%)", fontsize=9, color="#64748B")
        ax.tick_params(colors="#0D1B2A", labelsize=10)
        ax.spines[["top","right","bottom","left"]].set_visible(False)
        ax.xaxis.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.info("👈 Enter customer details on the left and click **Predict Churn** to see results.")

        # Feature importance chart (always visible)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Feature Importance (trained model)**")
        feat_names = ["Age", "Frequent Flyer", "Income Class", "Services Opted", "Acct Synced", "Booked Hotel"]
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)

        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor("#F4F8FB")
        ax.set_facecolor("#F4F8FB")
        colors = ["#028090" if i == sorted_idx[-1] else "#B2DFDB" for i in sorted_idx]
        ax.barh([feat_names[i] for i in sorted_idx],
                importances[sorted_idx], color=colors, edgecolor="none", height=0.6)
        ax.set_xlabel("Importance Score", fontsize=9, color="#64748B")
        ax.tick_params(colors="#0D1B2A", labelsize=9)
        ax.spines[["top","right","bottom"]].set_visible(False)
        ax.spines["left"].set_color("#E2E8F0")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small style='color:#94A3B8'>B.Tech Gen AI (2nd Semester) · Final Project · "
    "Random Forest Classifier · Deployed via Streamlit Cloud</small>",
    unsafe_allow_html=True
)
