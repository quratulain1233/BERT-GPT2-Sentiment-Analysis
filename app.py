
import streamlit as st
import torch
import pandas as pd
import html

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="SentimentIQ | AI Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BERT_ID = "qurrat99/bert-sentiment-analysis"
GPT2_ID = "qurrat99/gpt2-sentiment-analysis"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# 2. SESSION STATE
# =====================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

if "last_results" not in st.session_state:
    st.session_state.last_results = None

# =====================================================
# 3. THEME STYLING
# =====================================================

def apply_theme(theme):

    if theme == "Dark":

        bg = "#10121C"
        surface = "#191C2B"
        surface2 = "#202438"
        text = "#F5F5FA"
        muted = "#A4A9BB"
        border = "#30354A"
        accent = "#A78BFA"
        input_bg = "#191C2B"

    else:

        bg = "#F6F7FB"
        surface = "#FFFFFF"
        surface2 = "#F0EEFA"
        text = "#22243A"
        muted = "#777C91"
        border = "#E7E8F0"
        accent = "#7255D9"
        input_bg = "#FFFFFF"

    st.markdown(f"""
    <style>

    @import url(
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    :root {{
        --bg: {bg};
        --surface: {surface};
        --surface2: {surface2};
        --text: {text};
        --muted: {muted};
        --border: {border};
        --accent: {accent};
        --input: {input_bg};
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: var(--bg);
        color: var(--text);
    }}

    .block-container {{
        max-width: 1400px;
        padding: 2.2rem 3rem 3rem;
    }}

    [data-testid="stSidebar"] {{
        background: var(--surface);
        border-right: 1px solid var(--border);
    }}

    [data-testid="stSidebar"] .block-container {{
        padding: 2rem 1.3rem;
    }}

    h1, h2, h3, h4, p, label {{
        color: var(--text);
    }}

    .brand {{
        font-size: 27px;
        font-weight: 800;
        letter-spacing: -1px;
        color: var(--text);
    }}

    .brand span {{
        color: var(--accent);
    }}

    .brand-sub {{
        font-size: 11px;
        color: var(--muted);
        letter-spacing: 2px;
        margin-top: 3px;
    }}

    .hero {{
        background: linear-gradient(
            120deg,
            #25204C,
            #5140A0,
            #7660C9
        );
        padding: 43px;
        border-radius: 23px;
        margin-bottom: 28px;
        color: white;
        position: relative;
        overflow: hidden;
    }}

    .hero h1 {{
        color: white !important;
        font-size: 39px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 12px;
    }}

    .hero p {{
        color: #E4DDFD !important;
        font-size: 15px;
        max-width: 650px;
        line-height: 1.8;
    }}

    .eyebrow {{
        color: #D5C8FF;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2.5px;
        margin-bottom: 12px;
    }}

    .section-title {{
        font-size: 22px;
        font-weight: 750;
        color: var(--text);
        margin: 25px 0 5px;
    }}

    .section-sub {{
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 20px;
    }}

    .surface-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 23px;
        border-radius: 18px;
        margin-bottom: 16px;
    }}

    .metric-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 22px;
        border-radius: 17px;
        min-height: 120px;
    }}

    .metric-label {{
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.4px;
        color: var(--muted);
        text-transform: uppercase;
    }}

    .metric-value {{
        font-size: 28px;
        font-weight: 800;
        margin-top: 12px;
        color: var(--text);
    }}

    .metric-description {{
        font-size: 12px;
        color: var(--muted);
        margin-top: 5px;
    }}

    .result-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 19px;
        padding: 27px;
        text-align: center;
        min-height: 190px;
    }}

    .result-label {{
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
    }}

    .result-name {{
        font-size: 30px;
        font-weight: 800;
        margin: 17px 0 7px;
        color: var(--text);
    }}

    .result-confidence {{
        color: var(--accent);
        font-size: 24px;
        font-weight: 800;
    }}

    .stTextArea textarea {{
        background: var(--input) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 13px !important;
        padding: 15px !important;
        font-size: 15px !important;
    }}

    .stTextArea textarea:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }}

    .stButton > button {{
        background: #7255D9;
        color: white;
        border: none;
        border-radius: 11px;
        padding: 12px 20px;
        min-height: 46px;
        font-weight: 700;
        transition: 0.2s;
    }}

    .stButton > button:hover {{
        background: #5D42BD;
        color: white;
        transform: translateY(-1px);
    }}

    .stProgress > div > div > div > div {{
        background: var(--accent);
    }}

    div[data-baseweb="select"] > div {{
        background: var(--surface);
        border-color: var(--border);
    }}

    .sidebar-caption {{
        color: var(--muted);
        font-size: 11px;
        letter-spacing: 1.5px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 12px;
    }}

    .status-dot {{
        display: inline-block;
        height: 8px;
        width: 8px;
        background: #31B981;
        border-radius: 50%;
        margin-right: 7px;
    }}

    .footer {{
        text-align: center;
        padding: 25px;
        color: var(--muted);
        font-size: 12px;
        border-top: 1px solid var(--border);
        margin-top: 45px;
    }}

    </style>
    """, unsafe_allow_html=True)


apply_theme(st.session_state.theme)

# =====================================================
# 4. MODEL LOADING
# =====================================================

@st.cache_resource(show_spinner=False)
def load_model(model_id, model_type):

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id
    )

    if model_type == "GPT-2":
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# =====================================================
# 5. PREDICTION FUNCTION
# =====================================================

def predict_sentiment(text, model_id, model_type):

    tokenizer, model = load_model(
        model_id,
        model_type
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    predicted_id = torch.argmax(probabilities).item()

    raw_label = model.config.id2label.get(
        predicted_id,
        model.config.id2label.get(str(predicted_id), "")
    )

    label_mapping = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive",
        "0": "Negative",
        "1": "Neutral",
        "2": "Positive"
    }

    sentiment = label_mapping.get(
        str(raw_label).upper(),
        str(raw_label)
    )

    classes = ["Negative", "Neutral", "Positive"]

    result = {
        "sentiment": sentiment,
        "confidence": probabilities[predicted_id].item() * 100,
        "probabilities": {
            classes[i]: probabilities[i].item() * 100
            for i in range(len(probabilities))
        }
    }

    return result


# =====================================================
# 6. RESULT COMPONENT
# =====================================================

def show_result(result, model_name):

    sentiment = result["sentiment"]

    emoji = {
        "Positive": "😊",
        "Negative": "😞",
        "Neutral": "😐"
    }.get(sentiment, "✦")

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">{model_name.upper()} ANALYSIS</div>
        <div class="result-name">{emoji} {html.escape(sentiment)}</div>
        <div class="result-confidence">
            {result["confidence"]:.2f}%
        </div>
        <div style="color:var(--muted);font-size:12px;margin-top:5px;">
            Prediction confidence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("**Sentiment probabilities**")

    for label, value in result["probabilities"].items():

        st.markdown(
            f"<span style='color:var(--text);font-size:13px;'>"
            f"{label}</span>",
            unsafe_allow_html=True
        )

        st.progress(
            min(max(value / 100, 0.0), 1.0)
        )

        st.caption(f"{value:.2f}%")


# =====================================================
# 7. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("""
    <div class="brand">Sentiment<span>IQ.</span></div>
    <div class="brand-sub">AI INTELLIGENCE PLATFORM</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-caption">WORKSPACE</div>',
                unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Analyze Text",
            "Model Comparison",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown('<div class="sidebar-caption">APPEARANCE</div>',
                unsafe_allow_html=True)

    theme = st.selectbox(
        "Choose theme",
        ["Light", "Dark"],
        index=["Light", "Dark"].index(st.session_state.theme)
    )

    if theme != st.session_state.theme:
        st.session_state.theme = theme
        st.rerun()

    st.markdown("---")

    st.markdown("""
    <div class="sidebar-caption">MODEL ARCHITECTURES</div>
    <div style="font-size:13px;color:var(--text);line-height:2.4;">
    <span style="color:#7255D9;">●</span> BERT · Encoder-only
    <br>
    <span style="color:#A78BFA;">●</span> GPT-2 · Decoder-only
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.caption("SentimentIQ v1.0")


# =====================================================
# 8. DASHBOARD HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">INTELLIGENT TEXT ANALYTICS</div>
    <h1>Understand the emotion<br>behind every word.</h1>
    <p>
    An advanced sentiment intelligence workspace powered by
    two fine-tuned Transformer architectures: BERT and GPT-2.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 9. DASHBOARD PAGE
# =====================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">Your AI workspace</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-sub">'
        'Two architectures. One intelligent analysis platform.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-label">AI MODELS</div>
        <div class="metric-value">02</div>
        <div class="metric-description">Fine-tuned Transformer models</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-label">ARCHITECTURES</div>
        <div class="metric-value">02</div>
        <div class="metric-description">Encoder-only and Decoder-only</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-label">SENTIMENT CLASSES</div>
        <div class="metric-value">03</div>
        <div class="metric-description">Positive, Negative and Neutral</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Meet your models</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.markdown("""
        <div class="surface-card">
            <div style="font-size:35px;">◈</div>
            <h3>BERT</h3>
            <p style="color:var(--muted);">
            Bidirectional Encoder Representations from Transformers.
            </p>
            <hr>
            <b>Architecture</b>
            <p>Encoder-only Transformer</p>
            <b>Task</b>
            <p>Three-class sentiment classification</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="surface-card">
            <div style="font-size:35px;">✦</div>
            <h3>GPT-2</h3>
            <p style="color:var(--muted);">
            Generative Pre-trained Transformer 2.
            </p>
            <hr>
            <b>Architecture</b>
            <p>Decoder-only Transformer</p>
            <b>Task</b>
            <p>Three-class sentiment classification</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Start analyzing</div>',
                unsafe_allow_html=True)

    st.write("Choose an analysis workspace from the sidebar.")

    if st.button("Open Text Analysis →"):
        st.info("Select Analyze Text from the sidebar.")


# =====================================================
# 10. ANALYZE TEXT PAGE
# =====================================================

elif page == "Analyze Text":

    st.markdown(
        '<div class="section-title">Sentiment analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-sub">'
        'Enter a sentence, tweet or customer review and let AI analyze it.'
        '</div>',
        unsafe_allow_html=True
    )

    model_choice = st.selectbox(
        "Select your AI architecture",
        ["BERT", "GPT-2"]
    )

    architecture = (
        "Encoder-only · Bidirectional context"
        if model_choice == "BERT"
        else "Decoder-only · Causal attention"
    )

    st.caption(architecture)

    user_text = st.text_area(
        "Your text",
        placeholder="Write or paste your tweet here...",
        height=190,
        key="single_text"
    )

    st.caption(f"Characters: {len(user_text)}")

    if st.button("Analyze with AI ✦", use_container_width=True):

        if not user_text.strip():

            st.warning("Please enter some text first.")

        else:

            model_id = (
                BERT_ID if model_choice == "BERT"
                else GPT2_ID
            )

            with st.spinner(f"Analyzing with {model_choice}..."):

                try:

                    result = predict_sentiment(
                        user_text,
                        model_id,
                        model_choice
                    )

                    st.session_state.last_results = {
                        model_choice: result
                    }

                    st.markdown("---")
                    st.markdown("### Analysis result")

                    show_result(result, model_choice)

                except Exception as error:

                    st.error("Model prediction failed.")
                    st.exception(error)


# =====================================================
# 11. MODEL COMPARISON PAGE
# =====================================================

elif page == "Model Comparison":

    st.markdown(
        '<div class="section-title">Compare AI architectures</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-sub">'
        'Give both models the same text and compare their predictions.'
        '</div>',
        unsafe_allow_html=True
    )

    comparison_text = st.text_area(
        "Enter text for both models",
        placeholder="Example: The airline service was excellent!",
        height=160,
        key="comparison_text"
    )

    if st.button(
        "Compare BERT vs GPT-2 ✦",
        use_container_width=True
    ):

        if not comparison_text.strip():

            st.warning("Please enter text for comparison.")

        else:

            try:

                with st.spinner("Running both AI models..."):

                    bert_result = predict_sentiment(
                        comparison_text,
                        BERT_ID,
                        "BERT"
                    )

                    gpt2_result = predict_sentiment(
                        comparison_text,
                        GPT2_ID,
                        "GPT-2"
                    )

                st.session_state.last_results = {
                    "BERT": bert_result,
                    "GPT-2": gpt2_result
                }

                st.markdown("---")
                st.markdown("### Model predictions")

                col1, col2 = st.columns(2, gap="large")

                with col1:
                    show_result(bert_result, "BERT")

                with col2:
                    show_result(gpt2_result, "GPT-2")

                st.markdown("### Architecture comparison")

                comparison_df = pd.DataFrame({
                    "Model": ["BERT", "GPT-2"],
                    "Architecture": [
                        "Encoder-only",
                        "Decoder-only"
                    ],
                    "Predicted Sentiment": [
                        bert_result["sentiment"],
                        gpt2_result["sentiment"]
                    ],
                    "Confidence (%)": [
                        round(bert_result["confidence"], 2),
                        round(gpt2_result["confidence"], 2)
                    ]
                })

                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                    hide_index=True
                )

                if (
                    bert_result["sentiment"]
                    == gpt2_result["sentiment"]
                ):

                    st.success(
                        "Both models agree on the predicted sentiment."
                    )

                else:

                    st.warning(
                        "The models disagree on the predicted sentiment."
                    )

                st.caption(
                    "Confidence scores are model probabilities, "
                    "not proof of correctness."
                )

            except Exception as error:

                st.error("Comparison failed.")
                st.exception(error)


# =====================================================
# 12. ABOUT PROJECT PAGE
# =====================================================

elif page == "About Project":

    st.markdown(
        '<div class="section-title">About SentimentIQ</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="surface-card">
        <h3>Project overview</h3>
        <p style="color:var(--muted);line-height:1.9;">
        SentimentIQ is an AI-powered Natural Language Processing
        application designed to classify text into Positive,
        Negative and Neutral sentiment categories.
        It compares two different Transformer architectures
        fine-tuned on the same Twitter airline sentiment dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Technologies used")

    technologies = pd.DataFrame({
        "Technology": [
            "BERT",
            "GPT-2",
            "PyTorch",
            "Hugging Face Transformers",
            "Streamlit",
            "Python"
        ],
        "Purpose": [
            "Encoder-only architecture",
            "Decoder-only architecture",
            "Deep learning framework",
            "Model loading and fine-tuning",
            "Web application",
            "Programming language"
        ]
    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Classification categories")

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="surface-card">
        <h3>😊 Positive</h3>
        <p style="color:var(--muted);">Positive opinions and emotions.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="surface-card">
        <h3>😐 Neutral</h3>
        <p style="color:var(--muted);">Neutral or factual statements.</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="surface-card">
        <h3>😞 Negative</h3>
        <p style="color:var(--muted);">Negative opinions and emotions.</p>
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# 13. FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    <b>SentimentIQ</b> · AI Sentiment Intelligence Platform
    <br><br>
    Built with BERT, GPT-2, PyTorch, Hugging Face and Streamlit
</div>
""", unsafe_allow_html=True)
