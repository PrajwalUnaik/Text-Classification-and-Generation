# 🧠 NLP Interview Analyzer

Welcome to the **NLP Interview Analyzer** — an interactive web app built with Streamlit that classifies sports interview transcripts and generates intelligent answers using transformer-based models.

🔗 **Live App**: [Click to Open](https://text-classification-and-generationgit-g36pp9pnnqscjkfpjb95vx.streamlit.app/)

---

## ✨ Features

- 🗂️ **Transcript Classification**Automatically categorizes interview content into meaningful classes like `injury_update`, `post_game_reaction`, and more using a fine-tuned RoBERTa model.
- 💬 **Q&A Generator**Generates plausible, category-specific answers to user-entered questions using GPT-2.
- 📊 **Topic Cluster Visualization**
  View how interviews are grouped based on topic clusters with a UMAP + t-SNE embedding map.

---

## 🚀 Tech Stack

| Component            | Technology              |
| -------------------- | ----------------------- |
| Frontend             | Streamlit               |
| Classification Model | RoBERTa (fine-tuned)    |
| Text Generation      | GPT-2                   |
| Visualization        | Plotly + pandas + t-SNE |
| Hosting              | Streamlit Cloud         |

---

## 🧩 Project Structure

```
📁 Streamlit/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── label_map-2.pkl         # Category label encoder
├── inv_label_map-2.pkl     # Inverse label map
├── lda_tsne_embeddings.csv # UMAP/t-SNE cluster data
```

---

## 📦 Installation (Local)

```bash
git clone https://github.com/PrajwalUnaik/Text-Classification-and-Generation.git
cd Streamlit
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Model Sources

- 🔗 RoBERTa Model: [PrajwalUNaik/interview-nlp-models](https://huggingface.co/PrajwalUNaik/interview-nlp-models)
- 🤖 GPT-2: Preloaded via Hugging Face (`gpt2`)

---

## 🙋‍♂️ Creator

Built by **Prajwal UNaik** as part of a university NLP project at SRH University, combining advanced classification with interactive analytics.
