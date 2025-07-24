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
input_buffer = deque(maxlen=SEQ_LENGTH)
all_inputs = []        # Stores all user inputs
X_train = []           # Features
y_train = []           # Labels
session_data = []      # For display & export

# --- Labeling Logic ---
def classify(x):
    if x <= 1.5:
        return "Low"
    elif x <= 4.0:
        return "Medium"
    else:
        return "High"

# --- Feature Extraction ---
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
st.set_page_config(page_title="🧠 Aviator Pattern Learner", layout="centered")
st.title("🎯 Aviator Predictor: Live Pattern + Streak Learner")
st.markdown("""
Enter multipliers one by one.  
The app will **start predicting from the 11th round** after learning patterns from the first 10.
""")

# --- Input Section ---
new_input = st.number_input("🎲 Enter new multiplier (e.g., 1.25)", min_value=1.00, step=0.01, format="%.2f")

if st.button("➕ Submit Multiplier"):
    all_inputs.append(float(new_input))
    prediction = ""
    confidence = ""
    status = "🟡 Collecting initial 10 inputs..."

    if len(all_inputs) >= 11:
        # Train model using all previous data
        for i in range(len(all_inputs) - SEQ_LENGTH):
            window = all_inputs[i:i+SEQ_LENGTH]
            label = classify(all_inputs[i + SEQ_LENGTH])
            X_train.append(extract_features(window))
            y_train.append(label)

        # Latest window for prediction
        current_window = all_inputs[-SEQ_LENGTH:]
        features = extract_features(current_window)

        clf = GaussianNB()
        clf.fit(X_train, y_train)
        prob = clf.predict_proba([features])[0]
        pred_class = np.argmax(prob)
        prediction = clf.classes_[pred_class]
        confidence = prob[pred_class]

        if confidence < 0.6:
            status = f"⚠️ WAIT – Low confidence ({confidence*100:.2f}%)"
        else:
            status = f"✅ Prediction: **{prediction}** ({confidence*100:.2f}%)"

    elif len(all_inputs) < 11:
        status = f"🟡 Waiting for {11 - len(all_inputs)} more inputs to start predictions."

    # --- Save round to session log ---
    session_data.append({
        "Round": len(all_inputs),
        "Multiplier": float(new_input),
        "Prediction": prediction,
        "Confidence": f"{confidence*100:.2f}%" if confidence else "",
        "Status": status
    })

    st.success(f"✅ Recorded round {len(all_inputs)} → {new_input}")
    st.write(status)

# --- Show Session Table ---
if session_data:
    st.markdown("---")
    st.subheader("📄 Live Session Summary")
    df = pd.DataFrame(session_data)
    st.dataframe(df)

    # --- Manual Excel Export ---
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='SessionData')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="aviator_session.xlsx">📥 Download Session as Excel</a>'
    st.markdown(href, unsafe_allow_html=True)
