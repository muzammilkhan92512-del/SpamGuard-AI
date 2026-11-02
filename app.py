
import streamlit as st
import joblib
import pandas as pd
import csv
import io
import math

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="SpamGuard AI | Free Online SMS Spam Detector",
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

# ==================================================
# 2. SEO INFORMATION
# ==================================================

st.markdown("""
<meta name="description"
content="SpamGuard AI is a free online SMS spam detector.
Check suspicious messages, identify possible scam texts
and analyze SMS messages using machine learning.">

<meta name="keywords"
content="SMS spam detector, free spam checker, AI spam detector,
SMS scam checker, spam message detection, SpamGuard AI">

<meta name="robots" content="index, follow">
<meta name="author" content="SpamGuard AI">
""", unsafe_allow_html=True)

# Note: Streamlit may not place these tags in the actual
# document head. Search engines may therefore ignore them.
# A custom domain and proper HTML landing page are better
# for full SEO control.

# ==================================================
# 3. MODEL LOADING
# ==================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "spam_detector_improved.pkl is missing."
        )

    # Load only a trusted model artifact.
    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict):
        raise ValueError("Invalid model artifact.")

    if "model" not in artifact or "threshold" not in artifact:
        raise ValueError("Model information is missing.")

    model = artifact["model"]
    threshold = float(artifact["threshold"])

    if not hasattr(model, "predict_proba"):
        raise ValueError("Invalid prediction model.")

    if not hasattr(model, "classes_"):
        raise ValueError("Model classes are missing.")

    if "spam" not in list(model.classes_):
        raise ValueError("Spam class is missing.")

    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid spam threshold.")

    return model, threshold


try:
    model, SPAM_THRESHOLD = load_model()

except Exception:
    st.error("Unable to load SpamGuard AI model.")
    st.info(
        "Make sure spam_detector_improved.pkl "
        "is uploaded beside app.py."
    )
    st.stop()

# ==================================================
# 4. SESSION STATE
# ==================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

if "message_input" not in st.session_state:
    st.session_state.message_input = ""

# ==================================================
# 5. HELPER FUNCTIONS
# ==================================================

def sanitize_csv_value(value):

    value = str(value)

    if value.lstrip().startswith(
        ("=", "+", "-", "@", "\t", "\r")
    ):
        return "'" + value

    return value


def analyze_message(message):

    probabilities = model.predict_proba([message])[0]

    spam_index = list(model.classes_).index("spam")

    spam_probability = float(probabilities[spam_index])

    is_spam = spam_probability >= SPAM_THRESHOLD

    prediction = "SPAM" if is_spam else "NOT SPAM"

    confidence = (
        spam_probability if is_spam
        else 1 - spam_probability
    )

    return prediction, spam_probability, confidence


# ==================================================
# 6. PROFESSIONAL UI DESIGN
# ==================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(37, 99, 235, 0.16),
            transparent 35%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(8, 145, 178, 0.13),
            transparent 35%
        ),
        #080f1e;
    color: #f8fafc;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* HERO */

.hero {
    position: relative;
    overflow: hidden;
    padding: 48px 40px;
    border-radius: 24px;
    border: 1px solid rgba(96, 165, 250, 0.25);
    background: linear-gradient(
        120deg,
        rgba(30, 64, 175, 0.28),
        rgba(8, 145, 178, 0.17),
        rgba(15, 23, 42, 0.8)
    );
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
    animation: fadeUp 0.8s ease-out;
}

.hero::after {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    right: -80px;
    top: -100px;
    border-radius: 50%;
    background: rgba(56, 189, 248, 0.15);
    filter: blur(12px);
    animation: floatOrb 6s ease-in-out infinite;
}

.hero-tag {
    display: inline-block;
    padding: 8px 15px;
    color: #5eead4;
    border: 1px solid rgba(45, 212, 191, 0.4);
    border-radius: 30px;
    background: rgba(13, 148, 136, 0.12);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}

.hero-title {
    color: white;
    font-size: clamp(36px, 5vw, 58px);
    font-weight: 900;
    letter-spacing: -2px;
    margin-top: 15px;
    margin-bottom: 10px;
}

.hero-title span {
    background: linear-gradient(90deg, #60a5fa, #2dd4bf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    max-width: 760px;
    color: #bdcbe0;
    font-size: 16px;
    line-height: 1.8;
}

/* HEADINGS */

.section-heading {
    color: #f8fafc;
    font-size: 27px;
    font-weight: 850;
    margin-bottom: 5px;
}

.section-description {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.7;
    margin-bottom: 20px;
}

/* METRICS */

div[data-testid="stMetric"] {
    background: rgba(20, 35, 59, 0.85);
    border: 1px solid rgba(96, 165, 250, 0.2);
    border-radius: 17px;
    padding: 20px;
    transition: 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: #38bdf8;
    box-shadow: 0 10px 30px rgba(14,165,233,0.12);
}

div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: white;
    font-weight: 850;
}

/* INPUT */

.stTextArea textarea {
    background: #101d32 !important;
    color: white !important;
    border: 1px solid #334b6d !important;
    border-radius: 14px !important;
}

.stTextArea textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15);
}

/* BUTTONS */

.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(100deg, #2563eb, #0891b2);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 12px 18px;
    font-weight: 750;
    transition: 0.25s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(37,99,235,0.3);
}

/* CARDS */

.info-card {
    height: 100%;
    padding: 24px;
    border-radius: 17px;
    background: rgba(20, 35, 59, 0.78);
    border: 1px solid rgba(96, 165, 250, 0.2);
    transition: 0.25s ease;
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
    font-size: 14px;
    line-height: 1.8;
}

/* RESULTS */

.result-box {
    padding: 23px;
    border-radius: 15px;
    margin: 15px 0;
    font-size: 20px;
    font-weight: 850;
    animation: fadeUp 0.5s ease-out;
}

.spam {
    background: rgba(127,29,29,0.35);
    border: 1px solid #ef4444;
    color: #fecaca;
}

.safe {
    background: rgba(20,83,45,0.35);
    border: 1px solid #22c55e;
    color: #bbf7d0;
}

/* CONTACT */

.contact-link {
    display: block;
    text-decoration: none !important;
    color: #e0f2fe !important;
    background: rgba(30,64,175,0.16);
    border: 1px solid rgba(96,165,250,0.25);
    border-radius: 13px;
    padding: 17px;
    margin-top: 12px;
    transition: 0.25s ease;
}

.contact-link:hover {
    background: rgba(37,99,235,0.25);
    border-color: #38bdf8;
    transform: translateX(4px);
}

/* FOOTER */

.footer {
    text-align: center;
    padding: 30px 10px;
    color: #8192aa;
    font-size: 13px;
    line-height: 2;
    border-top: 1px solid rgba(148,163,184,0.15);
}

.footer strong {
    color: #e2e8f0;
}

/* ANIMATIONS */

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
        transform: translate(0,0);
    }
    50% {
        transform: translate(-25px,25px);
    }
}

/* MOBILE */

@media (max-width: 768px) {
    .hero {
        padding: 27px 20px;
    }

    .hero-title {
        font-size: 37px;
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

# ==================================================
# 7. HERO
# ==================================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">✦ INTELLIGENT MESSAGE PROTECTION</div>
    <div class="hero-title">🛡️ SpamGuard <span>AI</span></div>
    <div class="hero-subtitle">
        A free AI-powered SMS spam detector designed to help
        identify suspicious messages, unwanted texts and
        potential scam messages in seconds.
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# Navigation
n1, n2, n3, n4 = st.columns(4)

with n1:
    st.markdown("[🔍 Detector](#detector)")

with n2:
    st.markdown("[📊 History](#history)")

with n3:
    st.markdown("[✨ About](#about)")

with n4:
    st.markdown("[📞 Contact](#contact)")

st.divider()

# ==================================================
# 8. DASHBOARD
# ==================================================

total = len(st.session_state.history)

spam_count = sum(
    item["Prediction"] == "SPAM"
    for item in st.session_state.history
)

safe_count = total - spam_count

st.markdown("""
<div class="section-heading">Your Security Dashboard</div>
<div class="section-description">
Track your message analysis during this session.
</div>
""", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

m1.metric("Messages Analyzed", total)
m2.metric("Spam Detected", spam_count)
m3.metric("Safe Messages", safe_count)

# ==================================================
# 9. SPAM DETECTOR
# ==================================================

st.divider()

st.markdown("""
<div id="detector"></div>
<div class="section-heading">🔍 Free SMS Spam Checker</div>
<div class="section-description">
Paste your SMS message below to check whether it may be spam.
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.5, 1], gap="large")

with left:

    message = st.text_area(
        "Enter your SMS message",
        key="message_input",
        placeholder="Paste a suspicious SMS message here...",
        height=180,
        max_chars=MAX_MESSAGE_LENGTH
    )

    st.caption(
        f"Maximum length: {MAX_MESSAGE_LENGTH} characters."
    )

    detect_clicked = st.button(
        "🚀 Analyze Message",
        use_container_width=True
    )

    if detect_clicked:

        clean_message = (message or "").strip()

        if not clean_message:
            st.warning("Please enter a message first.")

        elif len(clean_message) > MAX_MESSAGE_LENGTH:
            st.error("Message is too long.")

        else:

            try:

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
                    "Unable to analyze this message. "
                    "Please try again."
                )

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
                "This message may be suspicious. "
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
            f"**Spam probability:** {spam_probability * 100:.2f}%"
        )

        st.write(
            f"**Prediction confidence:** {confidence * 100:.2f}%"
        )

        st.progress(confidence)

        st.caption(
            f"Decision threshold: {SPAM_THRESHOLD * 100:.0f}%. "
            "Predictions are estimates, not guarantees."
        )

with right:

    st.markdown("""
    <div class="info-card">
        <h3>⚡ How It Works</h3>
        <p>
            <b>01 — Enter Message</b><br>
            Paste any SMS you want to check.
        </p>
        <p>
            <b>02 — AI Analysis</b><br>
            Machine learning analyzes message patterns.
        </p>
        <p>
            <b>03 — Get Results</b><br>
            View the spam classification and confidence estimate.
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

# ==================================================
# 10. HISTORY
# ==================================================

st.divider()

st.markdown("""
<div id="history"></div>
<div class="section-heading">📊 Prediction History</div>
<div class="section-description">
Review and export messages analyzed in this session.
</div>
""", unsafe_allow_html=True)

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

    download_col, clear_col = st.columns(2)

    with download_col:

        st.download_button(
            "⬇️ Download CSV Report",
            csv_buffer.getvalue().encode("utf-8-sig"),
            "spam_detection_history.csv",
            "text/csv",
            use_container_width=True
        )

    with clear_col:

        if st.button(
            "🗑️ Clear History",
            use_container_width=True
        ):
            st.session_state.history = []
            st.session_state.last_analysis = None
            st.rerun()

else:

    st.info("Your analyzed messages will appear here.")

# ==================================================
# 11. ABOUT
# ==================================================

st.divider()

st.markdown("""
<div id="about"></div>
<div class="section-heading">✨ About SpamGuard AI</div>
<div class="section-description">
Smart SMS spam detection powered by machine learning.
</div>
""", unsafe_allow_html=True)

a1, a2, a3 = st.columns(3)

with a1:

    st.markdown("""
    <div class="info-card">
        <h3>🧠 Machine Learning</h3>
        <p>
            Word-level and character-level TF-IDF
            with Balanced Logistic Regression.
        </p>
    </div>
    """, unsafe_allow_html=True)

with a2:

    st.markdown("""
    <div class="info-card">
        <h3>⚡ Fast Analysis</h3>
        <p>
            Get a message classification and probability
            estimate within seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

with a3:

    st.markdown("""
    <div class="info-card">
        <h3>🔐 Privacy Awareness</h3>
        <p>
            Prediction history is held in the current session.
            Avoid submitting sensitive personal information.
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
    - ML library: Scikit-learn
    """)

    st.caption(
        "Model performance on new messages may differ from "
        "its held-out test results, especially for languages "
        "and message types not represented in its training data."
    )

# ==================================================
# 12. CONTACT AND SUPPORT
# ==================================================

st.divider()

st.markdown("""
<div id="contact"></div>
<div class="section-heading">📞 Contact & Support</div>
<div class="section-description">
Questions, suggestions or technical problems? Contact us.
</div>
""", unsafe_allow_html=True)

contact_left, contact_right = st.columns(2, gap="large")

with contact_left:

    st.markdown("""
    <div class="info-card">
        <h3>💬 Get in Touch</h3>
        <p>
            Contact the SpamGuard AI support team
            through email or WhatsApp.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <a class="contact-link" href="{EMAIL_LINK}">
            ✉️ <b>Email Support</b><br>
            <small>{CONTACT_EMAIL}</small>
        </a>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <a class="contact-link" href="{WHATSAPP_LINK}" target="_blank">
            🟢 <b>WhatsApp Support</b><br>
            <small>{PHONE_DISPLAY}</small>
        </a>
        """,
        unsafe_allow_html=True
    )

with contact_right:

    st.markdown("""
    <div class="info-card">
        <h3>🛠️ Feedback & Support</h3>
        <p>
            Report a problem, share an incorrect prediction,
            or suggest a new feature.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("support_form"):

        support_type = st.selectbox(
            "Support category",
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
            placeholder="Write your feedback...",
            max_chars=1500,
            height=120
        )

        submit_feedback = st.form_submit_button(
            "📨 Prepare Support Email",
            use_container_width=True
        )

        if submit_feedback:

            if not support_message.strip():

                st.warning("Please describe your issue.")

            else:

                subject = quote(
                    f"SpamGuard AI Support - {support_type}"
                )

                body = quote(
                    f"Support Category: {support_type}\n\n"
                    f"Message:\n{support_message}\n\n"
                    "Sent from SpamGuard AI."
                )

                mailto_url = (
                    f"mailto:{CONTACT_EMAIL}"
                    f"?subject={subject}&body={body}"
                )

                st.success(
                    "Your email draft has been prepared."
                )

                st.markdown(
                    f"[📧 Open Support Email]({mailto_url})"
                )

# ==================================================
# 13. PRIVACY
# ==================================================

st.divider()

with st.expander("🔒 Privacy & Security"):

    st.write("""
    - Messages are processed for spam classification.
    - Prediction history is held in the current session.
    - The application does not save messages to a permanent database.
    - The support form prepares an email draft; it does not
      automatically send or store support tickets.
    - Avoid submitting passwords, financial details,
      or other sensitive personal information.
    """)

# ==================================================
# 14. FOOTER
# ==================================================

st.markdown(
    f"""
    <div class="footer">
        <strong>🛡️ SpamGuard AI</strong><br>
        Free AI-Powered SMS Spam Detection<br><br>
        <a href="{EMAIL_LINK}"
           style="color:#60a5fa;text-decoration:none;">
           Email Support
        </a>
        &nbsp; | &nbsp;
        <a href="{WHATSAPP_LINK}" target="_blank"
           style="color:#5eead4;text-decoration:none;">
           WhatsApp
        </a>
        <br><br>
        © 2026 SpamGuard AI. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
