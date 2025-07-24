import streamlit as st
import pandas as pd
import numpy as np
from collections import deque
from sklearn.naive_bayes import GaussianNB
from io import BytesIO
import base64

# Config
SEQ_LENGTH = 10
CLASSES = ['Low (≤1.5x)', 'Medium (1.51x–4x)', 'High (>4x)']
input_buffer = deque(maxlen=SEQ_LENGTH)
session_data = []
X_train = []
y_train = []

# Classify multiplier
def classify(x):
    if x <= 1.5:
        return "Low"
    elif x <= 4.0:
        return "Medium"
    else:
        return "High"

# Extract features for training/prediction
def extract_features(seq):
    return [
        np.mean(seq),
        np.std(seq),
        seq[-1],                # most recent value
        min(seq),
        max(seq),
        seq.count(min(seq)),
        seq.count(max(seq)),
        sum(1 for x in seq if x <= 1.5),
        sum(1 for x in seq if x > 4.0)
    ]

# Streamlit UI
st.set_page_config(page_title="🧠 Aviator Predictor (Live Learning)", layout="centered")
st.title("🛫 Aviator Predictor — Live ML Learning (No Pre-trained Model)")
st.markdown("Manually enter the multiplier after each round. Starts learning and predicting after 10 entries.")

# Input form
new_value = st.number_input("🎲 Enter multiplier from latest round (e.g., 1.23)", min_value=1.00, step=0.01, format="%.2f")

if st.button("➕ Submit Multiplier"):
    input_buffer.append(float(new_value))

    prediction_status = "⏳ Not enough data yet for prediction."
    pred_label = ""
    confidence = ""

    if len(input_buffer) == SEQ_LENGTH:
        label = classify(float(new_value))
        y_train.append(label)
        features = extract_features(list(input_buffer))
        X_train.append(features)

        if len(X_train) >= 10:
            clf = GaussianNB()
            clf.fit(X_train[:-1], y_train[:-1])  # Train without the current input

            prob = clf.predict_proba([features])[0]
            pred_class = np.argmax(prob)
            pred_label = clf.classes_[pred_class]
            confidence = prob[pred_class]

            if confidence < 0.6:
                prediction_status = f"⚠️ WAIT — Low confidence ({confidence*100:.2f}%)"
            else:
                prediction_status = f"✅ Prediction: **{pred_label}** ({confidence*100:.2f}%)"
        else:
            prediction_status = f"📊 Learning... Need {10 - len(X_train)} more rounds to predict."

    # Save to session
    session_data.append({
        "Round": len(session_data) + 1,
        "Multiplier": float(new_value),
        "Prediction": pred_label,
        "Confidence": f"{confidence*100:.2f}%" if confidence else "",
        "Status": prediction_status
    })

    st.success(f"✅ Recorded round {len(session_data)} → {new_value}")
    st.write(prediction_status)

# Display session history
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
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="aviator_session.xlsx">📥 Download Session as Excel</a>'
    st.markdown(href, unsafe_allow_html=True)
