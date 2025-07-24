import streamlit as st
import pandas as pd
import numpy as np
from collections import deque
from sklearn.naive_bayes import GaussianNB
from io import BytesIO
import base64

# --- Config ---
SEQ_LENGTH = 10
CLASSES = ['Low (≤1.5x)', 'Medium (1.51x–4x)', 'High (>4x)']

# --- Buffers ---
all_inputs = []
X_train = []
y_train = []
session_data = []

# --- Label function ---
def classify(multiplier):
    if multiplier <= 1.5:
        return "Low"
    elif multiplier <= 4.0:
        return "Medium"
    else:
        return "High"

# --- Feature builder ---
def extract_features(seq):
    return [
        np.mean(seq),
        np.std(seq),
        seq[-1],
        min(seq),
        max(seq),
        seq.count(min(seq)),
        seq.count(max(seq)),
        sum(1 for x in seq if x <= 1.5),
        sum(1 for x in seq if x > 4.0)
    ]

# --- App UI ---
st.set_page_config(page_title="🧠 Aviator Predictor", layout="centered")
st.title("🎯 Aviator Predictor – Mobile Friendly")
st.markdown("Manually enter the multiplier after each round. Prediction starts after 10 inputs.")

# --- Input ---
multiplier = st.number_input("🎲 Enter multiplier (e.g. 2.15)", min_value=1.00, step=0.01, format="%.2f", key="user_input")

if st.button("➕ Submit"):
    all_inputs.append(float(multiplier))

    prediction = ""
    confidence = ""
    status = "⏳ Learning..."

    if len(all_inputs) >= SEQ_LENGTH + 1:
        # Build training data from history
        X_train.clear()
        y_train.clear()

        for i in range(len(all_inputs) - SEQ_LENGTH):
            window = all_inputs[i:i + SEQ_LENGTH]
            label = classify(all_inputs[i + SEQ_LENGTH])
            X_train.append(extract_features(window))
            y_train.append(label)

        if len(X_train) >= 5:
            model = GaussianNB()
            model.fit(X_train, y_train)

            current_window = all_inputs[-SEQ_LENGTH:]
            features = extract_features(current_window)
            probs = model.predict_proba([features])[0]
            pred_index = np.argmax(probs)
            prediction = model.classes_[pred_index]
            confidence = probs[pred_index]

            if confidence < 0.6:
                status = f"⚠️ WAIT — Low confidence ({confidence*100:.2f}%)"
            else:
                status = f"✅ Prediction: **{prediction}** ({confidence*100:.2f}%)"
        else:
            status = "📊 Collecting more training data..."
    else:
        status = f"🟡 Waiting for {SEQ_LENGTH + 1 - len(all_inputs)} more inputs to start predictions."

    # --- Save round log ---
    session_data.append({
        "Round": len(all_inputs),
        "Multiplier": multiplier,
        "Prediction": prediction,
        "Confidence": f"{confidence*100:.2f}%" if confidence else "",
        "Status": status
    })

    st.success(f"✅ Round {len(all_inputs)} recorded: {multiplier}")
    st.info(status)

# --- Show Table & Export ---
if session_data:
    st.markdown("---")
    st.subheader("📄 Session History")
    df = pd.DataFrame(session_data)
    st.dataframe(df)

    # Excel export
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='SessionData')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="aviator_session.xlsx">📥 Download Excel</a>'
    st.markdown(href, unsafe_allow_html=True)
