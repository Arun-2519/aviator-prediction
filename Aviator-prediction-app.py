
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# --- Page Setup ---
st.set_page_config(page_title="✈️ Aviator Signal AI", layout="centered")
st.title("✈️ Aviator Predictor – Live Pattern AI")
st.markdown("Enter multipliers one by one. AI starts predicting after 10 entries.")

# --- Session State ---
if "history" not in st.session_state:
    st.session_state.history = []
if "pred_log" not in st.session_state:
    st.session_state.pred_log = []

# --- Multiplier Classification ---
def classify(multiplier):
    if multiplier <= 1.5:
        return "Low"
    elif multiplier <= 4.0:
        return "Medium"
    else:
        return "High"

# --- Pattern Logic ---
def predict_signal(history):
    if len(history) < 10:
        return "⚠️ WAIT", 0.0, "Need 10 entries to begin prediction."
    
    # Count streaks
    last_5 = [classify(m) for m in history[-5:]]
    low_streak = sum([1 for x in last_5 if x == "Low"])
    medium_streak = sum([1 for x in last_5 if x == "Medium"])
    high_streak = sum([1 for x in last_5 if x == "High"])

    if low_streak >= 4:
        return "🔼 HIGH", 85.0, "Pattern: Spike often follows many LOWs."
    elif high_streak >= 3:
        return "⚠️ WAIT", 40.0, "High spike just occurred — likely dip."
    elif medium_streak >= 3:
        return "🟢 MEDIUM", 65.0, "Medium run forming — likely continuation."
    else:
        return "⚠️ WAIT", 50.0, "No strong pattern detected."

# --- UI ---
mult = st.number_input("🎲 Enter latest multiplier (e.g. 2.31)", min_value=0.0, format="%.2f", step=0.01)
add = st.button("✅ Submit")

if add and mult > 0:
    st.session_state.history.append(mult)
    st.success(f"✅ Recorded multiplier: {mult:.2f}")

# --- Prediction ---
if len(st.session_state.history) >= 10:
    signal, confidence, reason = predict_signal(st.session_state.history)
    st.markdown("## 📡 Prediction Engine")
    st.metric(label="🎯 Prediction", value=signal, delta=f"{confidence:.2f}%")
    st.caption(reason)

    result = st.selectbox("🔍 Enter actual result (Low/Medium/High):", ["Low", "Medium", "High"])
    if st.button("📥 Confirm & Learn"):
        st.session_state.pred_log.append({
            "Multiplier": mult,
            "Prediction": signal,
            "Confidence": f"{confidence:.2f}%",
            "Actual": result,
            "Correct": "✅" if result in signal else "❌"
        })
        st.rerun()
else:
    st.info(f"Waiting for {10 - len(st.session_state.history)} more inputs to begin prediction.")

# --- History Table ---
if st.session_state.pred_log:
    st.markdown("### 📄 Session Summary")
    df = pd.DataFrame(st.session_state.pred_log)
    st.dataframe(df)

    buf = BytesIO()
    df.to_excel(buf, index=False)
    st.download_button("⬇️ Download Prediction Log", data=buf.getvalue(), file_name="aviator_predictions.xlsx")
