import streamlit as st
import pandas as pd
import torch
import plotly.express as px
import joblib

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Tokenizer
)

# ============================
# Load Models and Artifacts
# ============================

@st.cache_resource
def load_classifier():
    model = RobertaForSequenceClassification.from_pretrained(
        "PrajwalUNaik/interview-nlp-models",
        subfolder="roberta-finetuned",
        use_safetensors=True
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        "PrajwalUNaik/interview-nlp-models",
        subfolder="roberta-finetuned",
        use_fast=False
    )
    return model, tokenizer

@st.cache_resource
def load_generator():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

@st.cache_data
def load_embeddings():
    return pd.read_csv("lda_tsne_embeddings.csv")

@st.cache_data
def load_label_map():
    return joblib.load("label_map-2.pkl"), joblib.load("inv_label_map-2.pkl")

# ============================
# Inference Functions
# ============================

def predict_category(text, model, tokenizer, inv_label_map):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return inv_label_map[predicted_label]

def generate_answer(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_new_tokens=60,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,  # Use greedy decoding for deterministic output
        num_beams=1,      # No beam search
        temperature=0.7,
        top_p=0.9
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the part after the prompt
    if prompt in decoded:
        answer = decoded.split(prompt)[-1].strip()
    else:
        answer = decoded.strip()
    # Post-process: stop at first line or punctuation
    for sep in [".", "\n", "Q:", "A:"]:
        if sep in answer:
            answer = answer.split(sep)[0].strip()
            break
    return answer

# ============================
# Streamlit UI
# ============================

st.set_page_config(page_title="NLP Interview Analyzer", layout="wide")

st.title("🧠 NLP Interview Analyzer")
st.markdown("""
Welcome to the **NLP Interview Analyzer**.  
This app helps you:

- 🗂️ Classify interview transcripts into categories.  
- 💬 Generate category-specific answers using AI.  
- 📊 Visualize topic clusters from the interview dataset.
""")

# Load models and data
model_clf, tokenizer_clf = load_classifier()
model_gen, tokenizer_gen = load_generator()
label_map, inv_label_map = load_label_map()
df_embed = load_embeddings()

# Section 1: Classification
st.markdown("---")
st.header("🗒️ Transcript Classification")
user_input = st.text_area("Paste the interview transcript here", placeholder="Start typing the transcript...")
if st.button("🔍 Predict Category") and user_input:
    label = predict_category(user_input, model_clf, tokenizer_clf, inv_label_map)
    st.success(f"📝 Predicted Category: **{label}**")

# Section 2: QA Generation
st.markdown("---")
st.header("💡 Category-Based Q&A Generator")
categories = list(label_map.keys())
selected_category = st.selectbox("Select a category", categories)
question = st.text_input("Enter your question")
if st.button("✨ Generate Answer") and question:
    prompt = f"Category: {selected_category}\nQ: {question}\nA:"
    response = generate_answer(prompt, model_gen, tokenizer_gen)
    st.info(response)

# Section 3: Visualization
st.markdown("---")
st.header("📈 Interactive Topic Embedding Visualization")
hover = st.checkbox("Show full text on hover", value=False)
hover_data = ["Interview Text"] if hover else ["Dominant_Topic"]
fig = px.scatter(df_embed, x="x", y="y", color="Dominant_Topic", hover_data=hover_data)
st.plotly_chart(fig, use_container_width=True)

st.caption("👤 Created by Prajwal UNaik")
