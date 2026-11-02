
import streamlit as st
import joblib
import pandas as pd
import csv
import io
import re
import math

from datetime import datetime, timezone
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "spam_detector_improved.pkl"

MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY = 100

# ==========================================
# 2. MODEL LOADING AND VALIDATION
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Improved model file is missing."
        )

    # Only load a trusted model file created by us.
    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict):
        raise ValueError("Invalid model artifact.")

    if "model" not in artifact or "threshold" not in artifact:
        raise ValueError("Required model information is missing.")

    model = artifact["model"]
    threshold = float(artifact["threshold"])

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support probability predictions.")

    if not hasattr(model, "classes_"):
        raise ValueError("Model classes are missing.")

    if "spam" not in list(model.classes_):
        raise ValueError("Spam class is missing from the model.")

    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid spam detection threshold.")

    return model, threshold


try:
    model, SPAM_THRESHOLD = load_model()

except Exception:
    st.error("⚠️ Unable to load SpamGuard AI model.")
    st.info(
        "Please check that spam_detector_improved.pkl "
        "is uploaded correctly beside app.py."
    )
    st.stop()


# ==========================================
# 3. SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None


# ==========================================
# 4. SECURITY HELPERS
# ==========================================

def sanitize_csv_value(value):
    """
    Prevent spreadsheet formula injection
    when users open exported CSV files.
    """

    value = str(value)

    if value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value

    return value


def analyze_message(message):

    probabilities = model.predict_proba([message])[0]

    spam_index = list(model.classes_).index("spam")

    spam_probability = float(probabilities[spam_index])

    is_spam = spam_probability >= SPAM_THRESHOLD

    prediction = "SPAM" if is_spam else "NOT SPAM"

    confidence = (
        spam_probability
        if is_spam
        else 1 - spam_probability
    )

    return prediction, spam_probability, confidence


# ==========================================
# 5. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background: #0b1120;
    color: #f8fafc;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
}

.subtitle {
    color: #94a3b8;
    font-size: 16px;
}

div[data-testid="stMetric"] {
    background: #172238;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263650;
}

.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 10px 22px;
    font-weight: 700;
}

.result-box {
    padding: 22px;
    border-radius: 12px;
    margin-top: 15px;
    font-size: 20px;
    font-weight: bold;
}

.spam {
    background: #451a1a;
    border: 1px solid #ef4444;
    color: #fecaca;
}

.safe {
    background: #123524;
    border: 1px solid #22c55e;
    color: #bbf7d0;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 6. HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🛡️ SpamGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Advanced SMS Spam Detection powered by Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 7. DASHBOARD METRICS
# ==========================================

total = len(st.session_state.history)

spam_count = sum(
    item["Prediction"] == "SPAM"
    for item in st.session_state.history
)

safe_count = total - spam_count

c1, c2, c3 = st.columns(3)

c1.metric("Messages Analyzed", total)
c2.metric("Spam Detected", spam_count)
c3.metric("Safe Messages", safe_count)

st.divider()


# ==========================================
# 8. MESSAGE DETECTOR
# ==========================================

left, right = st.columns([1.5, 1])

with left:

    st.subheader("🔍 Analyze a Message")

    message = st.text_area(
        "Enter your SMS message",
        placeholder="Paste a suspicious message here...",
        height=170,
        max_chars=MAX_MESSAGE_LENGTH
    )

    st.caption(
        f"Maximum message length: {MAX_MESSAGE_LENGTH} characters."
    )

    detect_clicked = st.button(
        "🚀 Detect Spam",
        use_container_width=True
    )

    if detect_clicked:

        if not message or not message.strip():

            st.warning("Please enter a message.")

        elif len(message) > MAX_MESSAGE_LENGTH:

            st.error("Message is too long.")

        else:

            try:

                prediction, spam_probability, confidence = (
                    analyze_message(message.strip())
                )

                timestamp = datetime.now(
                    timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S UTC")

                st.session_state.history.insert(0, {
                    "Time": timestamp,
                    "Message": message.strip(),
                    "Prediction": prediction,
                    "Spam Probability": round(
                        spam_probability * 100, 2
                    ),
                    "Confidence": round(
                        confidence * 100, 2
                    )
                })

                # Limit stored history per session.
                st.session_state.history = (
                    st.session_state.history[:MAX_HISTORY]
                )

                if prediction == "SPAM":

                    st.markdown(
                        '<div class="result-box spam">'
                        '⚠️ SPAM MESSAGE DETECTED'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.warning(
                        "This message has suspicious characteristics. "
                        "Avoid clicking unknown links or sharing "
                        "personal information."
                    )

                else:

                    st.markdown(
                        '<div class="result-box safe">'
                        '✅ MESSAGE CLASSIFIED AS NOT SPAM'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.success(
                        "The model classified this message as legitimate."
                    )

                st.write(
                    f"**Spam probability:** "
                    f"{spam_probability * 100:.2f}%"
                )

                st.write(
                    f"**Prediction confidence:** "
                    f"{confidence * 100:.2f}%"
                )

                st.progress(confidence)

                st.caption(
                    f"Spam decision threshold: "
                    f"{SPAM_THRESHOLD * 100:.0f}%. "
                    "Model estimates are not guarantees."
                )

            except Exception:
                st.error(
                    "Unable to analyze this message. "
                    "Please try again."
                )


# ==========================================
# 9. ABOUT SECTION
# ==========================================

with right:

    st.subheader("✨ About SpamGuard AI")

    st.info("""
    **Improved Machine Learning Model**

    - Word-level TF-IDF
    - Character-level TF-IDF
    - Balanced Logistic Regression
    - Validation-based threshold tuning
    - SMS Spam Collection dataset
    - Real-time predictions
    - Input validation
    - Protected CSV exports
    """)

    st.metric(
        "Spam Decision Threshold",
        f"{SPAM_THRESHOLD * 100:.0f}%"
    )

    st.subheader("💡 Test Messages")

    st.code(
        "Congratulations! You won a free iPhone. "
        "Click here to claim your prize!"
    )

    st.code(
        "Hey, are you coming to university tomorrow?"
    )

    st.code(
        "congratulation you win iPhone claim for contact "
        "on this number 923201292512"
    )


# ==========================================
# 10. PREDICTION HISTORY
# ==========================================

st.divider()

st.subheader("📊 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Message": st.column_config.TextColumn(
                "Message",
                width="large"
            )
        }
    )

    # Protect exported CSV values against formulas.
    safe_df = history_df.copy()

    for column in safe_df.columns:
        safe_df[column] = safe_df[column].apply(
            sanitize_csv_value
        )

    csv_buffer = io.StringIO()

    safe_df.to_csv(
        csv_buffer,
        index=False,
        quoting=csv.QUOTE_ALL
    )

    st.download_button(
        "⬇️ Download CSV Report",
        csv_buffer.getvalue().encode("utf-8-sig"),
        "spam_detection_history.csv",
        "text/csv",
        use_container_width=True
    )

    if st.button("🗑️ Clear History"):

        st.session_state.history = []

        st.rerun()

else:

    st.info("No messages analyzed yet.")


# ==========================================
# 11. PRIVACY NOTICE
# ==========================================

st.divider()

st.subheader("🔒 Privacy & Security")

st.caption(
    "Messages are processed for spam classification. "
    "Prediction history is held in the current Streamlit session "
    "and is not saved by this application to a permanent database. "
    "Avoid submitting passwords, financial details, or other "
    "sensitive personal information."
)


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "SpamGuard AI | Python • Scikit-learn • TF-IDF • Streamlit"
)
