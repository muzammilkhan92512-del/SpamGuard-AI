
import streamlit as st
import joblib
import pandas as pd
import csv
import io
import math

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SpamGuard AI | Smart Message Protection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "spam_detector_improved.pkl"

MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY = 100

CONTACT_EMAIL = "muzammilexprt@gmail.com"
PHONE_DISPLAY = "03201292512"
WHATSAPP_LINK = "https://wa.me/923201292512"
EMAIL_LINK = f"mailto:{CONTACT_EMAIL}"

# ==========================================
# 2. MODEL LOADING
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Improved model file is missing.")

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
        raise ValueError("Spam class is missing.")

    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid spam detection threshold.")

    return model, threshold


try:
    model, SPAM_THRESHOLD = load_model()

except Exception:
    st.error("⚠️ Unable to load SpamGuard AI model.")
    st.info(
        "Check that spam_detector_improved.pkl "
        "is uploaded beside app.py."
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
# 4. HELPER FUNCTIONS
# ==========================================

def sanitize_csv_value(value):
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
        spam_probability if is_spam else 1 - spam_probability
    )

    return prediction, spam_probability, confidence


# ==========================================
# 5. PROFESSIONAL ANIMATED DESIGN
# ==========================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background:
        radial-gradient(ellipse at 10% 10%, #172554 0%, transparent 38%),
        radial-gradient(ellipse at 90% 20%, #164e63 0%, transparent 35%),
        linear-gradient(135deg, #070d1b, #0b1427 55%, #101b32);
    color: #f8fafc;
}

/* Hide default Streamlit header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Animated hero */
.hero {
    position: relative;
    overflow: hidden;
    padding: 38px 35px;
    border: 1px solid rgba(96, 165, 250, 0.25);
    border-radius: 22px;
    background: linear-gradient(
        120deg,
        rgba(30, 64, 175, 0.25),
        rgba(15, 118, 110, 0.16),
        rgba(30, 41, 59, 0.5)
    );
    box-shadow: 0 15px 55px rgba(0,0,0,0.25);
    animation: fadeUp 0.8s ease-out;
}

.hero:after {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    right: -65px;
    top: -100px;
    border-radius: 50%;
    background: rgba(56, 189, 248, 0.12);
    filter: blur(8px);
    animation: floatOrb 6s ease-in-out infinite;
}

.hero-title {
    color: #ffffff;
    font-size: clamp(34px, 5vw, 55px);
    font-weight: 900;
    letter-spacing: -1.8px;
    margin-bottom: 8px;
}

.hero-title span {
    background: linear-gradient(90deg, #60a5fa, #2dd4bf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #bdcbe0;
    font-size: 17px;
    line-height: 1.7;
    max-width: 750px;
}

.hero-tag {
    display: inline-block;
    padding: 7px 13px;
    border: 1px solid rgba(45, 212, 191, 0.4);
    border-radius: 30px;
    color: #5eead4;
    background: rgba(13, 148, 136, 0.12);
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* Section headings */
.section-heading {
    color: #f8fafc;
    font-size: 27px;
    font-weight: 800;
    margin-bottom: 5px;
}

.section-description {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 20px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(20, 35, 59, 0.82);
    border: 1px solid rgba(96, 165, 250, 0.19);
    border-radius: 16px;
    padding: 19px;
    transition: all 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: #38bdf8;
    box-shadow: 0 10px 28px rgba(14, 165, 233, 0.12);
}

div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f8fafc;
    font-weight: 800;
}

/* Input fields */
.stTextArea textarea {
    background: rgba(15, 29, 50, 0.95) !important;
    color: #f8fafc !important;
    border: 1px solid #334b6d !important;
    border-radius: 13px !important;
    transition: border 0.2s ease, box-shadow 0.2s ease;
}

.stTextArea textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.15);
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(100deg, #2563eb, #0891b2);
    color: white !important;
    border: 0;
    border-radius: 11px;
    padding: 11px 18px;
    font-weight: 750;
    transition: all 0.25s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.3);
    border: 0;
}

/* Result boxes */
.result-box {
    padding: 22px;
    border-radius: 14px;
    margin: 15px 0;
    font-size: 19px;
    font-weight: 800;
    animation: fadeUp 0.45s ease-out;
}

.spam {
    background: rgba(127, 29, 29, 0.35);
    border: 1px solid #ef4444;
    color: #fecaca;
}

.safe {
    background: rgba(20, 83, 45, 0.35);
    border: 1px solid #22c55e;
    color: #bbf7d0;
}

/* Information cards */
.info-card {
    height: 100%;
    background: rgba(20, 35, 59, 0.75);
    border: 1px solid rgba(96, 165, 250, 0.2);
    border-radius: 16px;
    padding: 23px;
    transition: all 0.25s ease;
}

.info-card:hover {
    transform: translateY(-4px);
    border-color: #38bdf8;
}

.info-card h3 {
    color: #e0f2fe;
    font-size: 19px;
}

.info-card p {
    color: #a9b9cf;
    line-height: 1.7;
    font-size: 14px;
}

/* Contact links */
.contact-link {
    display: block;
    text-decoration: none !important;
    background: rgba(30, 64, 175, 0.16);
    border: 1px solid rgba(96, 165, 250, 0.25);
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
    color: #e0f2fe !important;
    transition: all 0.2s ease;
}

.contact-link:hover {
    background: rgba(37, 99, 235, 0.25);
    border-color: #38bdf8;
    transform: translateX(4px);
}

/* Footer */
.footer {
    text-align: center;
    padding: 28px 10px;
    color: #8192aa;
    font-size: 13px;
    border-top: 1px solid rgba(148, 163, 184, 0.15);
}

.footer strong {
    color: #cbd5e1;
}

/* Animations */
@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes floatOrb {
    0%, 100% {
        transform: translate(0, 0);
    }
    50% {
        transform: translate(-25px, 25px);
    }
}

/* Mobile */
@media (max-width: 768px) {
    .hero {
        padding: 25px 20px;
    }

    .hero-subtitle {
        font-size: 14px;
    }

    .section-heading {
        font-size: 23px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 6. HERO HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">✦ AI-POWERED MESSAGE SECURITY</div>
    <div class="hero-title">🛡️ SpamGuard <span>AI</span></div>
    <div class="hero-subtitle">
        Protect your inbox with intelligent SMS spam detection.
        Analyze suspicious messages instantly using machine learning
        and make smarter decisions about your messages.
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# Navigation
nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    st.markdown("[🔍 Detector](#detector)")

with nav2:
    st.markdown("[📊 History](#history)")

with nav3:
    st.markdown("[ℹ️ About](#about)")

with nav4:
    st.markdown("[📞 Contact](#contact)")

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

st.markdown(
    '<div class="section-heading">Your Security Dashboard</div>'
    '<div class="section-description">'
    'A quick overview of your current analysis session.'
    '</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

c1.metric("Messages Analyzed", total)
c2.metric("Spam Detected", spam_count)
c3.metric("Safe Messages", safe_count)

st.write("")


# ==========================================
# 8. MESSAGE DETECTOR
# ==========================================

st.markdown(
    '<div id="detector"></div>'
    '<div class="section-heading">🔍 Smart Message Detector</div>'
    '<div class="section-description">'
    'Paste an SMS message below and let AI analyze it.'
    '</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1.5, 1], gap="large")

with left:

    st.text_area(
        "Your SMS message",
        key="message_input",
        placeholder=(
            "Example: Congratulations! You won a free prize. "
            "Click here to claim..."
        ),
        height=180,
        max_chars=MAX_MESSAGE_LENGTH
    )

    st.caption(
        f"Maximum message length: {MAX_MESSAGE_LENGTH} characters."
    )

    detect_clicked = st.button(
        "🚀 Analyze Message",
        use_container_width=True
    )

    if detect_clicked:

        message = st.session_state.message_input or ""

        if not message.strip():

            st.warning("Please enter a message first.")

        elif len(message) > MAX_MESSAGE_LENGTH:

            st.error("Message is too long.")

        else:

            try:

                clean_message = message.strip()

                prediction, spam_probability, confidence = (
                    analyze_message(clean_message)
                )

                timestamp = datetime.now(
                    timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S UTC")

                st.session_state.history.insert(0, {
                    "Time": timestamp,
                    "Message": clean_message,
                    "Prediction": prediction,
                    "Spam Probability": round(
                        spam_probability * 100, 2
                    ),
                    "Confidence": round(
                        confidence * 100, 2
                    )
                })

                st.session_state.history = (
                    st.session_state.history[:MAX_HISTORY]
                )

                st.session_state.last_analysis = {
                    "prediction": prediction,
                    "spam_probability": spam_probability,
                    "confidence": confidence
                }

            except Exception:
                st.error(
                    "Unable to analyze this message. Please try again."
                )

    # Show the latest result
    if st.session_state.last_analysis:

        result = st.session_state.last_analysis

        prediction = result["prediction"]
        spam_probability = result["spam_probability"]
        confidence = result["confidence"]

        if prediction == "SPAM":

            st.markdown(
                '<div class="result-box spam">'
                '⚠️ SPAM MESSAGE DETECTED'
                '</div>',
                unsafe_allow_html=True
            )

            st.warning(
                "This message may be suspicious. Avoid clicking "
                "unknown links or sharing personal information."
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
            f"**Spam probability:** {spam_probability * 100:.2f}%"
        )

        st.write(
            f"**Prediction confidence:** {confidence * 100:.2f}%"
        )

        st.progress(confidence)

        st.caption(
            f"Spam decision threshold: {SPAM_THRESHOLD * 100:.0f}%. "
            "Predictions are estimates, not guarantees."
        )


with right:

    st.markdown("""
    <div class="info-card">
        <h3>⚡ How It Works</h3>
        <p>
            <b>1. Enter your message</b><br>
            Paste any SMS you want to check.
        </p>
        <p>
            <b>2. AI analyzes the text</b><br>
            Our machine learning model checks message patterns.
        </p>
        <p>
            <b>3. Get your result</b><br>
            Instantly see whether the message is classified as spam.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.metric(
        "Spam Decision Threshold",
        f"{SPAM_THRESHOLD * 100:.0f}%"
    )

    with st.expander("Try sample messages"):

        st.code(
            "Congratulations! You won a free iPhone. "
            "Click here to claim your prize!"
        )

        st.code(
            "Hey, are you coming to university tomorrow?"
        )

        st.code(
            "You have won a cash prize. Call now to claim."
        )


# ==========================================
# 9. PREDICTION HISTORY
# ==========================================

st.divider()

st.markdown(
    '<div id="history"></div>'
    '<div class="section-heading">📊 Analysis History</div>'
    '<div class="section-description">'
    'Review and export the messages analyzed during this session.'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

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

    col_download, col_clear = st.columns(2)

    with col_download:
        st.download_button(
            "⬇️ Download CSV Report",
            csv_buffer.getvalue().encode("utf-8-sig"),
            "spam_detection_history.csv",
            "text/csv",
            use_container_width=True
        )

    with col_clear:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_analysis = None
            st.rerun()

else:

    st.info("No messages analyzed yet. Your results will appear here.")


# ==========================================
# 10. ABOUT SECTION
# ==========================================

st.divider()

st.markdown(
    '<div id="about"></div>'
    '<div class="section-heading">✨ About SpamGuard AI</div>'
    '<div class="section-description">'
    'Machine learning technology designed to help identify suspicious SMS messages.'
    '</div>',
    unsafe_allow_html=True
)

a1, a2, a3 = st.columns(3)

with a1:
    st.markdown("""
    <div class="info-card">
        <h3>🧠 Machine Learning</h3>
        <p>
            Uses word-level and character-level TF-IDF features
            with Logistic Regression to classify SMS messages.
        </p>
    </div>
    """, unsafe_allow_html=True)

with a2:
    st.markdown("""
    <div class="info-card">
        <h3>⚡ Instant Analysis</h3>
        <p>
            Submit a message and receive a classification
            and confidence estimate within seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

with a3:
    st.markdown("""
    <div class="info-card">
        <h3>🔐 Privacy Awareness</h3>
        <p>
            Analysis history stays in the current application
            session. Avoid submitting sensitive personal information.
        </p>
    </div>
    """, unsafe_allow_html=True)


with st.expander("Technical information"):

    st.write("""
    - Model: Balanced Logistic Regression
    - Features: Word TF-IDF and Character TF-IDF
    - Dataset: SMS Spam Collection
    - Threshold: Validation-based threshold
    - Framework: Python and Streamlit
    - Machine Learning: Scikit-learn
    """)

    st.caption(
        "The model was evaluated on a held-out dataset. "
        "Performance on new messages may differ, particularly "
        "for languages or message types not represented in its training data."
    )


# ==========================================
# 11. CONTACT AND SUPPORT
# ==========================================

st.divider()

st.markdown(
    '<div id="contact"></div>'
    '<div class="section-heading">📞 Contact & Support</div>'
    '<div class="section-description">'
    'Have a question, found a problem, or want to share feedback? '
    'Get in touch.'
    '</div>',
    unsafe_allow_html=True
)

contact_left, contact_right = st.columns(2, gap="large")

with contact_left:

    st.markdown("""
    <div class="info-card">
        <h3>💬 Contact Us</h3>
        <p>
            Need assistance with SpamGuard AI? Contact us
            through email or WhatsApp.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<a class="contact-link" href="{EMAIL_LINK}">'
        f'✉️ &nbsp; <b>Email Support</b><br>'
        f'<small>{CONTACT_EMAIL}</small></a>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<a class="contact-link" href="{WHATSAPP_LINK}" target="_blank">'
        f'🟢 &nbsp; <b>WhatsApp Support</b><br>'
        f'<small>{PHONE_DISPLAY}</small></a>',
        unsafe_allow_html=True
    )

with contact_right:

    st.markdown("""
    <div class="info-card">
        <h3>🛠️ Submit Feedback</h3>
        <p>
            Tell us about a bug, incorrect prediction,
            or improvement you would like to see.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("support_form"):

        support_type = st.selectbox(
            "What do you need help with?",
            [
                "Report a bug",
                "Incorrect spam prediction",
                "General question",
                "Feature suggestion",
                "Other feedback"
            ]
        )

        support_message = st.text_area(
            "Describe your issue",
            placeholder="Write your feedback here...",
            max_chars=1500,
            height=120
        )

        submit_feedback = st.form_submit_button(
            "📨 Prepare Support Email",
            use_container_width=True
        )

        if submit_feedback:

            if not support_message.strip():

                st.warning("Please describe your issue first.")

            else:

                subject = quote(
                    f"SpamGuard AI Support - {support_type}"
                )

                body = quote(
                    f"Support Type: {support_type}\n\n"
                    f"Message:\n{support_message}\n\n"
                    f"Sent from SpamGuard AI."
                )

                mailto_url = (
                    f"mailto:{CONTACT_EMAIL}"
                    f"?subject={subject}&body={body}"
                )

                st.success(
                    "Your email draft is ready. "
                    "Use the link below to open it in your email application."
                )

                st.markdown(
                    f"[📧 Open Support Email]({mailto_url})"
                )


# ==========================================
# 12. PRIVACY NOTICE
# ==========================================

st.divider()

with st.expander("🔒 Privacy & Security"):

    st.write("""
    - Messages are processed for spam classification.
    - Prediction history is held in the current Streamlit session.
    - This application does not save messages to a permanent database.
    - The support form prepares an email in your email application;
      it does not automatically send or store support tickets.
    - Avoid submitting passwords, financial details, or other
      sensitive personal information.
    """)


# ==========================================
# 13. FOOTER
# ==========================================

st.markdown(f"""
<div class="footer">
    <strong>🛡️ SpamGuard AI</strong><br>
    Intelligent SMS Spam Detection powered by Machine Learning<br><br>
    <a href="{EMAIL_LINK}" style="color:#60a5fa;text-decoration:none;">
        Contact Support
    </a>
    &nbsp; | &nbsp;
    <a href="{WHATSAPP_LINK}" target="_blank"
       style="color:#5eead4;text-decoration:none;">
        WhatsApp
    </a>
    <br><br>
    © 2026 SpamGuard AI. All rights reserved.
</div>
""", unsafe_allow_html=True)
