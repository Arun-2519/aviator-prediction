import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from collections import deque
from io import BytesIO
import base64

# --- Session State Initialization ---
if "multipliers" not in st.session_state:
    st.session_state.multipliers = []
if "X_train" not in st.session_state:
    st.session_state.X_train = []
if "y_train" not in st.session_state:
    st.session_state.y_train = []
if "log" not in st.session_state:
    st.session_state.log = []
if "streak" not in st.session_state:
    st.session_state.streak = 0

SEQ_LENGTH = 10
MIN_TRAIN = 20
CONF_THRESHOLD = 0.65

# --- Helper Functions ---
def classify(mult):
    if mult <= 1.5:
        return "Low"
    elif mult <= 4.0:
        return "Medium"
    return "High"

def extract_features(seq):
    return [
        np.mean(seq),
        np.std(seq),
        seq[-1],
        min(seq),
        max(seq),
        sum(1 for x in seq if x <= 1.5),
        sum(1 for x in seq if x > 4.0),
        len(set(seq))  # uniqueness
    ]

# --- UI ---
st.set_page_config(page_title="✈️ Aviator Predictor AI", layout="centered")
st.title("✈️ Aviator Predictor – Live Pattern AI")
st.markdown("Enter each round's multiplier below. Predictions start after 10 inputs.")

# --- Input Section ---
mult = st.number_input("🎲 Enter latest multiplier (e.g. 2.31)", min_value=1.0, step=0.01, format="%.2f")
if st.button("➕ Submit Multiplier"):
    st.session_state.multipliers.append(mult)
    st.success(f"✅ Round {len(st.session_state.multipliers)} recorded → {mult}")

# --- Train model if 10+ entries ---
if len(st.session_state.multipliers) > SEQ_LENGTH:
    st.markdown("---")
    st.subheader("📡 Prediction Engine")

    # Train model on past data
    st.session_state.X_train.clear()
    st.session_state.y_train.clear()

    for i in range(len(st.session_state.multipliers) - SEQ_LENGTH):
        window = st.session_state.multipliers[i:i+SEQ_LENGTH]
        label = classify(st.session_state.multipliers[i+SEQ_LENGTH])
        features = extract_features(window)
        st.session_state.X_train.append(features)
        st.session_state.y_train.append(label)

    if len(st.session_state.X_train) >= 5:
        clf = GaussianNB()
        clf.fit(st.session_state.X_train, st.session_state.y_train)

        current_window = st.session_state.multipliers[-SEQ_LENGTH:]
        current_features = extract_features(current_window)
        probs = clf.predict_proba([current_features])[0]
        pred_index = np.argmax(probs)
        pred_class = clf.classes_[pred_index]
        conf = probs[pred_index]

        if conf >= CONF_THRESHOLD:
            st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)
            st.success(f"🎯 Prediction: **{pred_class}** | Confidence: {conf*100:.2f}%")
        else:
            st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)
            st.warning(f"⚠️ WAIT — Low confidence ({conf*100:.2f}%)")

        # --- Learn from actual input ---
        actual = st.selectbox("🔍 Enter actual result:", ["Low", "Medium", "High"])
        if st.button("✅ Confirm & Learn"):
            correct = actual == pred_class
            st.session_state.log.append({
                "Prediction": pred_class,
                "Confidence": f"{conf*100:.2f}%",
                "Actual": actual,
                "Result": "✅" if correct else "❌"
            })

            if correct:
                st.session_state.streak = 0
            else:
                st.session_state.streak += 1

            st.success(f"🔁 Learning complete. Added actual: {actual}")
            st.session_state.multipliers.append(mult)
            st.rerun()
    else:
        st.info("📊 Collecting more data to train...")

else:
    needed = SEQ_LENGTH + 1 - len(st.session_state.multipliers)
    st.info(f"📥 Please enter {needed} more multipliers to begin predictions.")

# --- History ---
if st.session_state.log:
    st.markdown("---")
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df)

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    link = f'<a href="data:application/octet-stream;base64,{b64}" download="aviator_history.xlsx">📥 Download Excel</a>'
    st.markdown(link, unsafe_allow_html=True)

st.caption("🤖 Built with Streamlit + Naive Bayes. Pattern-aware, confidence-based, and fully adaptive.")
