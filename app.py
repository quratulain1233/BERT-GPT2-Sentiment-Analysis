
import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sentiment AI | Smart Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_ID = "qurrat99/bert-sentiment-analysis"

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020 0%, #15132e 55%, #101b35 100%);
    color: #f8fafc;
}

header {background: transparent !important;}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #312e81, #6d28d9, #2563eb);
    padding: 38px;
    border-radius: 22px;
    margin-bottom: 28px;
    box-shadow: 0 15px 45px rgba(0,0,0,0.25);
}

.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 8px;
}

.hero p {
    color: #e0e7ff;
    font-size: 16px;
}

.panel {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 24px;
    border-radius: 18px;
    margin-bottom: 20px;
}

.result-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
}

.result-card h2 {
    font-size: 30px;
    margin: 10px 0;
}

.small-label {
    color: #a5b4fc;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.stTextArea textarea {
    background: #171d35 !important;
    color: white !important;
    border: 1px solid #55518a !important;
    border-radius: 14px !important;
    font-size: 16px !important;
}

.stButton button {
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
    font-weight: 700;
    width: 100%;
    min-height: 48px;
}

.stButton button:hover {
    background: linear-gradient(90deg, #6d28d9, #1d4ed8);
    color: white;
    border: 1px solid #a78bfa;
}

section[data-testid="stSidebar"] {
    background: #10162b;
    border-right: 1px solid #30365a;
}

footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🧠 Sentiment AI")
    st.caption("Intelligent Text Analysis")
    st.divider()

    st.markdown("### About the model")
    st.write("**Architecture:** BERT")
    st.write("**Task:** Sentiment Classification")
    st.write("**Classes:** Positive, Negative, Neutral")

    st.divider()
    st.markdown("### How to use")
    st.write("1. Enter a tweet or sentence.")
    st.write("2. Click Analyze Sentiment.")
    st.write("3. View the prediction and confidence.")

    st.divider()
    st.caption("Powered by Hugging Face Transformers")


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="small-label">ARTIFICIAL INTELLIGENCE • NLP</div>
    <h1>🧠 Sentiment Intelligence</h1>
    <p>Understand emotions behind words with your fine-tuned BERT model.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- MODEL STATUS ----------------
try:
    with st.spinner("Connecting to your BERT model..."):
        tokenizer, model = load_model()
    st.success("● BERT model connected and ready")
except Exception as e:
    st.error("Unable to load the model. Check your Hugging Face repository.")
    st.exception(e)
    st.stop()


# ---------------- INPUT SECTION ----------------
left, right = st.columns([1.35, 0.85], gap="large")

with left:
    st.markdown("### ✍️ Enter your text")

    user_text = st.text_area(
        "Text to analyze",
        placeholder="Example: I absolutely love the service. The experience was amazing!",
        height=190,
        label_visibility="collapsed"
    )

    analyze = st.button("✨ Analyze Sentiment", use_container_width=True)

with right:
    st.markdown("""
    <div class="panel">
        <div class="small-label">YOUR AI ASSISTANT</div>
        <h2>Discover the emotion.</h2>
        <p>Analyze customer feedback, tweets, reviews and everyday text.</p>
        <hr>
        <p>🟢 Positive sentiment</p>
        <p>🔴 Negative sentiment</p>
        <p>🟡 Neutral sentiment</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- PREDICTION ----------------
if analyze:
    if not user_text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing your text..."):
            inputs = tokenizer(
                user_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)[0]

            predicted_id = torch.argmax(probabilities).item()
            sentiment = model.config.id2label[predicted_id]
            confidence = probabilities[predicted_id].item() * 100

        normalized = sentiment.upper()

        if "POSITIVE" in normalized:
            emoji = "😊"
            color = "#22c55e"
            description = "This text expresses a positive emotion."
        elif "NEGATIVE" in normalized:
            emoji = "😞"
            color = "#f87171"
            description = "This text expresses a negative emotion."
        else:
            emoji = "😐"
            color = "#facc15"
            description = "This text expresses a neutral emotion."

        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="small-label">PREDICTED SENTIMENT</div>
                <h2>{emoji} {normalized}</h2>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="small-label">MODEL CONFIDENCE</div>
                <h2>{confidence:.2f}%</h2>
                <p>Confidence in the predicted class</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📈 Sentiment probability breakdown")

        labels = [
            model.config.id2label[i]
            for i in range(len(probabilities))
        ]

        chart_data = pd.DataFrame({
            "Sentiment": labels,
            "Probability (%)": [
                p.item() * 100 for p in probabilities
            ]
        }).set_index("Sentiment")

        st.bar_chart(chart_data, horizontal=True)

        st.markdown(
            f"<p style='color:{color};font-weight:bold;'>"
            f"Analysis completed successfully.</p>",
            unsafe_allow_html=True
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#94a3b8;'>"
    "Sentiment Intelligence | Built with BERT, PyTorch & Streamlit"
    "</p>",
    unsafe_allow_html=True
)
