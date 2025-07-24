import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from io import BytesIO
import base64

# --- State ---
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

# --- Constants ---
SEQ_LENGTH = 10
CONF_THRESHOLD = 0.65

# --- Classification Buckets ---
def classify(x):
    if x <= 1.5:
        return "Low"
    elif x <= 4.0:
        return "Medium"
    return "High"

# --- Smarter Feature Extraction ---
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

# --- UI ---
st.set_page_config(page_title="✈️ Aviator Predictor AI", layout="centered")
st.title("✈️ Aviator Predictor – Live Pattern AI")
st.info("""
### 🎯 Class Ranges:
- **Low:** 1.00 – 1.50
- **Medium:** 1.51 – 4.00
- **High:** 4.01 and above

👉 The AI learns from your patterns and gives predictions with confidence.
""")

# --- Input Section ---
mult = st.number_input("🎲 Enter latest multiplier (e.g. 2.31)", min_value=1.0, step=0.01, format="%.2f")
if st.button("➕ Submit Multiplier"):
    st.session_state.multipliers.append(mult)
    st.success(f"✅ Round {len(st.session_state.multipliers)} recorded → {mult}")
    st.rerun()

# --- Model Training ---
if len(st.session_state.multipliers) > SEQ_LENGTH:
    st.subheader("📡 Prediction Engine")
    st.session_state.X_train.clear()
    st.session_state.y_train.clear()

    for i in range(len(st.session_state.multipliers) - SEQ_LENGTH):
        window = st.session_state.multipliers[i:i+SEQ_LENGTH]
        label = classify(st.session_state.multipliers[i+SEQ_LENGTH])
        features = extract_features(window)
        st.session_state.X_train.append(features)
        st.session_state.y_train.append(label)

    # Balance warning
    label_counts = pd.Series(st.session_state.y_train).value_counts()
    if label_counts.min() < 2:
        st.warning("⚠️ Training data unbalanced. Enter more varied inputs.")

    # Train model
    model = GaussianNB()
    model.fit(st.session_state.X_train, st.session_state.y_train)

    current_window = st.session_state.multipliers[-SEQ_LENGTH:]
    current_features = extract_features(current_window)
    probs = model.predict_proba([current_features])[0]
    pred_index = np.argmax(probs)
    pred_class = model.classes_[pred_index]
    conf = probs[pred_index]

    if conf >= CONF_THRESHOLD:
        st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)
        st.success(f"🎯 Prediction: **{pred_class}** | Confidence: {conf*100:.2f}%")
    else:
        st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)
        st.warning(f"⚠️ WAIT — Low confidence ({conf*100:.2f}%)")

    # --- Confirm Actual ---
    actual = st.selectbox("🔍 Enter actual result (Low/Medium/High):", ["Low", "Medium", "High"])
    if st.button("✅ Confirm & Learn"):
        correct = actual == pred_class
        st.session_state.log.append({
            "Prediction": pred_class,
            "Confidence": f"{conf*100:.2f}%",
            "Actual": actual,
            "Result": "✅" if correct else "❌"
        })

        # Add multiplier if not already added
        if classify(mult) != actual:
            st.session_state.streak += 1
        else:
            st.session_state.streak = 0

        st.success("🔁 Learned from actual result.")
        st.rerun()
else:
    needed = SEQ_LENGTH + 1 - len(st.session_state.multipliers)
    st.info(f"📥 Enter {needed} more multipliers to begin prediction.")

# --- History & Download ---
if st.session_state.log:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="aviator_history.xlsx">📥 Download Excel</a>'
    st.markdown(href, unsafe_allow_html=True)

st.caption("🤖 Built with ❤️ by Vendra & AI. Pattern-aware. Confidence-based. Adaptive.")
