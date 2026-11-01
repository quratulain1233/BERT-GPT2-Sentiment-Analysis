
import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Sentiment Intelligence AI",
    page_icon="🧠",
    layout="wide"
)

BERT_ID = "qurrat99/bert-sentiment-analysis"
GPT2_ID = "qurrat99/gpt2-sentiment-analysis"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #17132f, #101b35);
    color: white;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}

.hero {
    background: linear-gradient(120deg, #312e81, #6d28d9, #2563eb);
    padding: 38px;
    border-radius: 22px;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 38px;
}

.hero p {
    color: #e0e7ff;
}

.panel {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 24px;
    border-radius: 18px;
    margin-bottom: 18px;
}

.result-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
}

.result-card h2 {
    font-size: 27px;
}

.small-label {
    color: #a5b4fc;
    font-size: 13px;
    letter-spacing: 2px;
}

.stTextArea textarea {
    background: #171d35 !important;
    color: white !important;
    border-radius: 12px !important;
}

.stButton button {
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 12px;
    min-height: 48px;
    font-weight: bold;
}

.stButton button:hover {
    color: white;
    border: 1px solid #a78bfa;
}

section[data-testid="stSidebar"] {
    background: #10162b;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_model(model_id, model_type):

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id
    )

    # GPT-2 requires a padding token
    if model_type == "GPT-2":
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ==========================================
# SENTIMENT PREDICTION
# ==========================================

def predict_sentiment(text, model_id, model_type):

    tokenizer, model = load_model(model_id, model_type)

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

    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive",
        "0": "Negative",
        "1": "Neutral",
        "2": "Positive"
    }

    sentiment = label_map.get(
        str(raw_label).upper(),
        str(raw_label)
    )

    confidence = probabilities[predicted_id].item() * 100

    result = {
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": {
            label: probabilities[i].item() * 100
            for i, label in enumerate(
                ["Negative", "Neutral", "Positive"]
            )
        }
    }

    return result


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🧠 Sentiment AI")

    st.caption("Dual Transformer Intelligence")

    st.divider()

    st.markdown("### Available Models")

    st.write("🔵 BERT – Encoder-only")
    st.write("🟣 GPT-2 – Decoder-only")

    st.divider()

    st.markdown("### About the Project")

    st.write("""
    This application compares two Transformer
    architectures for three-class sentiment
    classification.
    """)

    st.divider()

    st.caption("Powered by Hugging Face and Streamlit")


# ==========================================
# MAIN HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <div class="small-label">ARTIFICIAL INTELLIGENCE • NLP</div>
    <h1>🧠 Sentiment Intelligence AI</h1>
    <p>
    Analyze emotions using two powerful Transformer
    architectures: BERT and GPT-2.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# MODEL SELECTION
# ==========================================

st.markdown("### ⚙️ Select Your AI Model")

model_choice = st.radio(
    "Choose analysis mode:",
    ["BERT", "GPT-2", "Compare Both"],
    horizontal=True
)

if model_choice == "BERT":
    st.info("Using BERT – Encoder-only Transformer.")

elif model_choice == "GPT-2":
    st.info("Using GPT-2 – Decoder-only Transformer.")

else:
    st.info("Both models will analyze the same text.")


# ==========================================
# TEXT INPUT
# ==========================================

st.markdown("### ✍️ Enter Your Text")

user_text = st.text_area(
    "Tweet or sentence",
    placeholder="Example: I absolutely love this airline. Amazing service!",
    height=150,
    label_visibility="collapsed"
)

analyze = st.button(
    "✨ Analyze Sentiment",
    use_container_width=True
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

def show_result(result, model_name):

    sentiment = result["sentiment"]
    confidence = result["confidence"]

    emoji = {
        "Positive": "😊",
        "Negative": "😞",
        "Neutral": "😐"
    }.get(sentiment, "🧠")

    st.markdown(f"""
    <div class="result-card">
        <div class="small-label">{model_name} PREDICTION</div>
        <h2>{emoji} {sentiment}</h2>
        <h3>{confidence:.2f}%</h3>
        <p>Prediction confidence</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Probability Breakdown")

    probability_df = pd.DataFrame({
        "Sentiment": list(result["probabilities"].keys()),
        "Probability (%)": list(result["probabilities"].values())
    }).set_index("Sentiment")

    st.bar_chart(probability_df)


# ==========================================
# RUN ANALYSIS
# ==========================================

if analyze:

    if not user_text.strip():

        st.warning("Please enter some text first.")

    else:

        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        try:

            if model_choice == "BERT":

                with st.spinner("Analyzing with BERT..."):

                    result = predict_sentiment(
                        user_text,
                        BERT_ID,
                        "BERT"
                    )

                show_result(result, "BERT")

            elif model_choice == "GPT-2":

                with st.spinner("Analyzing with GPT-2..."):

                    result = predict_sentiment(
                        user_text,
                        GPT2_ID,
                        "GPT-2"
                    )

                show_result(result, "GPT-2")

            else:

                with st.spinner("Comparing BERT and GPT-2..."):

                    bert_result = predict_sentiment(
                        user_text,
                        BERT_ID,
                        "BERT"
                    )

                    gpt2_result = predict_sentiment(
                        user_text,
                        GPT2_ID,
                        "GPT-2"
                    )

                col1, col2 = st.columns(2)

                with col1:
                    show_result(bert_result, "BERT")

                with col2:
                    show_result(gpt2_result, "GPT-2")

                st.markdown("### 🔍 Model Comparison")

                comparison = pd.DataFrame({
                    "Model": ["BERT", "GPT-2"],
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
                    comparison,
                    use_container_width=True,
                    hide_index=True
                )

                if bert_result["sentiment"] == gpt2_result["sentiment"]:
                    st.success("Both models predicted the same sentiment.")
                else:
                    st.warning("The models predicted different sentiments.")

        except Exception as error:

            st.error("An error occurred during prediction.")
            st.exception(error)


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
<p style="text-align:center;color:#94a3b8;">
Sentiment Intelligence AI | BERT + GPT-2
<br>
Built with PyTorch, Hugging Face Transformers and Streamlit
</p>
""", unsafe_allow_html=True)
