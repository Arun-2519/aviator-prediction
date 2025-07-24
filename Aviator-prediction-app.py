
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from io import BytesIO
import base64

# --- State Initialization ---
if "multipliers" not in st.session_state:
    st.session_state.multipliers = []
if "X_train" not in st.session_state:
    st.session_state.X_train = []
if "y_train" not in st.session_state:
    st.session_state.y_train = []
if "log" not in st.session_state:
    st.session_state.log = []

SEQ_LENGTH = 10

# --- Feature Extraction ---
def extract_features(seq):
    log_seq = [np.log(x + 1) for x in seq]
    return [
        np.mean(log_seq),
        np.std(log_seq),
        seq[-1],
        max(seq),
        min(seq),
        sum(1 for x in seq if x > 4.0) / len(seq),
        len(set(seq))
    ]

# --- UI Setup ---
st.set_page_config(page_title="✈️ Aviator Predictor AI", layout="centered")
st.title("✈️ Aviator Predictor – Advanced ML Edition")

st.info("Enter multipliers after each round. Predictions start after 10 inputs.")

mult = st.number_input("🎲 Enter multiplier (e.g. 2.31)", min_value=1.0, step=0.01, format="%.2f")

if st.button("➕ Submit Multiplier"):
    st.session_state.multipliers.append(mult)
    st.success(f"✅ Recorded multiplier: {mult}")
    st.rerun()

# --- Learning and Prediction ---
if len(st.session_state.multipliers) > SEQ_LENGTH:
    st.session_state.X_train.clear()
    st.session_state.y_train.clear()

    for i in range(len(st.session_state.multipliers) - SEQ_LENGTH):
        window = st.session_state.multipliers[i:i+SEQ_LENGTH]
        target = st.session_state.multipliers[i+SEQ_LENGTH]
        st.session_state.X_train.append(extract_features(window))
        st.session_state.y_train.append(target)

    model = GradientBoostingRegressor()
    model.fit(st.session_state.X_train, st.session_state.y_train)

    current_window = st.session_state.multipliers[-SEQ_LENGTH:]
    current_features = extract_features(current_window)
    predicted_value = model.predict([current_features])[0]

    st.success(f"🎯 Predicted next multiplier: **{predicted_value:.2f}x**")

    actual = st.number_input("📌 Enter actual multiplier result (to improve learning)", min_value=1.0, step=0.01, format="%.2f")
    if st.button("✅ Confirm Actual Result"):
        st.session_state.log.append({
            "Prediction": round(predicted_value, 2),
            "Actual": actual,
            "Error": round(abs(actual - predicted_value), 2)
        })
        st.session_state.multipliers.append(actual)
        st.success("🧠 Learned from this new data.")
        st.rerun()

else:
    st.warning(f"Need {SEQ_LENGTH + 1 - len(st.session_state.multipliers)} more inputs to start prediction.")

# --- History and Download ---
if st.session_state.log:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="aviator_prediction_history.xlsx">📥 Download Excel</a>'
    st.markdown(href, unsafe_allow_html=True)

st.caption("🤖 Built using Gradient Boosting + Pattern Extraction + Real Value Learning")
