# app.py
import streamlit as st
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

@st.cache_resource
def load_tfidf():
    return joblib.load("models/tfidf_logreg.joblib")

@st.cache_resource
def load_distilbert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "models/distilbert_model"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    return tok, model, device

st.title("📰 Fake News Detection Bot")
model_choice = st.radio("Model", ["TF-IDF + Logistic Regression (fast)", "DistilBERT (accurate)"], horizontal=True)

text = st.text_area("Paste a news statement/article:", height=180, placeholder="Type or paste the content here...")

if st.button("Analyse"):
    if not text.strip():
        st.warning("Please paste some text.")
        st.stop()

    if model_choice.startswith("TF-IDF"):
        clf = load_tfidf()
        pred = clf.predict([text])[0]
        prob = max(clf.predict_proba([text])[0])
        label = "Real ✅" if pred == 1 else "Fake ❌"
        st.subheader(label)
        st.caption(f"Confidence: {prob:.2f}")
    else:
        tok, model, device = load_distilbert()
        encoded = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
        prob = float(probs[pred])
        label = "Real ✅" if pred == 1 else "Fake ❌"
        st.subheader(label)
        st.caption(f"Confidence: {prob:.2f}")