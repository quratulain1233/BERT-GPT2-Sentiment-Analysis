
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "qurrat99/bert-sentiment-analysis"

st.set_page_config(
    page_title="BERT Sentiment Analysis",
    page_icon="💬"
)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model

st.title("💬 BERT Sentiment Analysis")
st.write("Analyze the sentiment of your tweets using your trained BERT model.")

with st.spinner("Loading BERT model..."):
    tokenizer, model = load_model()

st.success("BERT model loaded successfully!")

text = st.text_area(
    "Enter your tweet:",
    placeholder="I love this airline! Its service is amazing."
)

if st.button("Analyze Sentiment"):
    if text.strip():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]

        prediction = torch.argmax(probabilities).item()
        label = model.config.id2label[prediction]

        st.subheader("Prediction")
        st.success(f"Sentiment: {label}")
        st.write(
            f"Confidence: {probabilities[prediction].item() * 100:.2f}%"
        )

        st.subheader("All class probabilities")

        for i, probability in enumerate(probabilities):
            st.write(
                f"{model.config.id2label[i]}: "
                f"{probability.item() * 100:.2f}%"
            )
    else:
        st.warning("Please enter a tweet.")
