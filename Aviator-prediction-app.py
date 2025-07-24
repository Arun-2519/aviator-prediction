import streamlit as st
import pandas as pd
import numpy as np
from collections import deque
from sklearn.naive_bayes import GaussianNB
from io import BytesIO
import base64

# --- Settings ---
SEQ_LENGTH = 10
CLASSES = ['Low (≤1.5x)', 'Medium (1.51x–4x)', 'High (>4x)']

# --- Global buffers ---
all_inputs = []        # All user multipliers
X_train = []           # ML features
y_train = []           # ML labels
session_data = []      # Display and Excel download

# --- Labeling ---
def classify(x):
    if x <= 1.5:
        return "Low"
    elif x <= 4.0:
        return "Medium"
    else:
        return "High"

# --- Feature Engineering ---
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

# --- Streamlit UI ---
st.set_page_config(page_title="🧠 Aviator Predictor (Live)", layout="centered")
st.title("🎯 Aviator Live Predictor – Pattern & Streak Learner")
st.markdown("Manually enter the multiplier after each round. Prediction starts from the 11th entry automatically.")

# --- Input form using key to trigger on ENTER ---
multiplier = st.number_input("🎲 Enter multiplier and press **ENTER**", key="multiplier_input", min_value=1.00, step=0.01, format="%.2f")

# Auto-detect if new input (user pressed ENTER)
if "last_value" not in st.session_state:
    st.session_state.last_value = None

if multiplier != st.session_state.last_value:
    st.session_state.last_value = multiplier
    all_inputs.append(float(multiplier))

    prediction = ""
    confidence = ""
    status = "🟡 Collecting initial 10 inputs..."

    if len(all_inputs) >= 11:
        # Train from previous SEQ_LENGTHs
        X_train.clear()
        y_train.clear()
        for i in range(len(all_inputs) - SEQ_LENGTH):
            window = all_inputs[i:i + SEQ_LENGTH]
            label = classify(all_inputs[i + SEQ_LENGTH])
            X_train.append(extract_features(window))
            y_train.append(label)

        current_window = all_inputs[-SEQ_LENGTH:]
        features = extract_features(current_window)

        clf = GaussianNB()
        clf.fit(X_train, y_train)
        prob = clf.predict_proba([features])[0]
        pred_class = np.argmax(prob)
        prediction = clf.classes_[pred_class]
        confidence = prob[pred_class]

        if confidence < 0.6:
            status = f"⚠️ WAIT — Low confidence ({confidence*100:.2f}%)"
        else:
            status = f"✅ Prediction: **{prediction}** ({confidence*100:.2f}%)"
    else:
        status = f"🟡 Waiting for {11 - len(all_inputs)} more inputs to start predictions."

    # Save session round
    session_data.append({
        "Round": len(all_inputs),
        "Multiplier": float(multiplier),
        "Prediction": prediction,
        "Confidence": f"{confidence*100:.2f}%" if confidence else "",
        "Status": status
    })

    st.success(f"✅ Recorded Round {len(all_inputs)} → {multiplier}")
    st.info(status)

# --- Display Session Table ---
if session_data:
    st.markdown("---")
    st.subheader("📄 Session Summary")
    df = pd.DataFrame(session_data)
    st.dataframe(df)

    # Manual Excel download
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='SessionData')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="aviator_session.xlsx">📥 Download Excel</a>'
    st.markdown(href, unsafe_allow_html=True)
