
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "spam_detector_improved.pkl"

# ---------------------------------
# LOAD IMPROVED MODEL
# ---------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Upload spam_detector_improved.pkl to GitHub."
        )

    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict):
        raise ValueError(
            "Invalid model format. Expected a dictionary."
        )

    if "model" not in artifact or "threshold" not in artifact:
        raise ValueError(
            "Model file must contain 'model' and 'threshold' keys. "
            "Upload the correct improved model."
        )

    model = artifact["model"]
    threshold = float(artifact["threshold"])

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Loaded model does not support probability predictions."
        )

    return model, threshold


try:
    model, SPAM_THRESHOLD = load_model()

except Exception as e:
    st.error("⚠️ Model Loading Error")
    st.code(str(e))
    st.info(
        "Check that spam_detector_improved.pkl is uploaded "
        "in the same GitHub folder as app.py."
    )
    st.stop()

# ---------------------------------
# SESSION STATE
# ---------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------
# CUSTOM DESIGN
# ---------------------------------

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

# ---------------------------------
# HEADER
# ---------------------------------

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

# ---------------------------------
# DASHBOARD METRICS
# ---------------------------------

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

# ---------------------------------
# MESSAGE DETECTOR
# ---------------------------------

left, right = st.columns([1.5, 1])

with left:

    st.subheader("🔍 Analyze a Message")

    message = st.text_area(
        "Enter your SMS message",
        placeholder="Paste a suspicious message here...",
        height=170
    )

    if st.button("🚀 Detect Spam", use_container_width=True):

        if not message.strip():
            st.warning("Please enter a message.")

        else:
            try:
                probabilities = model.predict_proba([message])[0]

                spam_index = list(model.classes_).index("spam")

                spam_probability = float(
                    probabilities[spam_index]
                )

                is_spam = spam_probability >= SPAM_THRESHOLD

                result = "SPAM" if is_spam else "NOT SPAM"

                confidence = (
                    spam_probability
                    if is_spam
                    else 1 - spam_probability
                )

                st.session_state.history.insert(0, {
                    "Time": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "Message": message,
                    "Prediction": result,
                    "Spam Probability": round(
                        spam_probability * 100, 2
                    ),
                    "Confidence": round(
                        confidence * 100, 2
                    )
                })

                if is_spam:

                    st.markdown(
                        '<div class="result-box spam">'
                        '⚠️ SPAM MESSAGE DETECTED'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.warning(
                        "This message has suspicious characteristics. "
                        "Avoid clicking links or sharing personal information."
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
                    "Probabilities are model estimates, not guarantees."
                )

            except Exception as e:
                st.error(f"Prediction Error: {e}")

# ---------------------------------
# ABOUT SECTION
# ---------------------------------

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

# ---------------------------------
# PREDICTION HISTORY
# ---------------------------------

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

    csv = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV Report",
        csv,
        "spam_detection_history.csv",
        "text/csv"
    )

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

else:

    st.info("No messages analyzed yet.")

# ---------------------------------
# FOOTER
# ---------------------------------

st.divider()

st.caption(
    "SpamGuard AI | Python • Scikit-learn • TF-IDF • Streamlit"
)
