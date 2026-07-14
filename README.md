
# Customer Feedback Intelligence Platform

An end-to-end NLP pipeline that fine-tunes transformer models on 18,000+ real Amazon product reviews, builds a semantic search engine, generates automated insights using LLMs, and presents everything in an interactive live dashboard.

**Live Demo:** [customer-feedback-intel.streamlit.app](https://customer-feedback-intel.streamlit.app)

---

## What it does

- **Sentiment Analysis** — classifies reviews as positive/negative using DistilBERT + LoRA (89.91% accuracy)
- **Aspect Detection** — identifies which product dimensions are mentioned (price, quality, delivery, packaging, usability, customer service)
- **Topic Clustering** — groups 18,188 reviews into 19 meaningful topics using BERTopic
- **Semantic Search** — search reviews in plain English with FAISS + cross-encoder reranking (MRR = 1.0)
- **LLM Insights** — auto-generates cluster summaries, trend reports, and weekly insight documents using Groq (Llama 3)
- **Interactive Dashboard** — 5-page Streamlit app with charts, search, and real-time review analysis

---

## Results

| Model | Metric | Score |
|-------|--------|-------|
| Sentiment (DistilBERT + LoRA) | Accuracy | **89.91%** |
| Sentiment (DistilBERT + LoRA) | F1 Weighted | **0.8991** |
| Sentiment (DistilBERT + LoRA) | Trainable Params | **1.09%** |
| Aspect (DistilBERT + LoRA) | F1 Micro | 0.3503 |
| Aspect (DistilBERT + LoRA) | F1 Macro | 0.3201 |
| Topics (BERTopic) | Topics Found | 19 |
| Topics (BERTopic) | Topic Diversity | 0.9579 |
| Retrieval (FAISS + Reranker) | MRR | **1.0000** |
| Retrieval (FAISS + Reranker) | Hit Rate @5 | **1.0000** |
| Retrieval (FAISS + Reranker) | Latency P50 | **215ms** |

---

## Architecture

```
Amazon Reviews (18,188)
        │
        ▼
┌─────────────────┐
│  Stage 1        │  Data pipeline: download → clean → split → validate
│  Data Pipeline  │
└────────┬────────┘
         │ train/val/test parquet (80/10/10 stratified)
         ▼
┌─────────────────┐
│  Stage 2        │  DistilBERT + LoRA (sentiment, 89.91% accuracy)
│  Fine-tuning    │  DistilBERT + LoRA (aspects, weighted BCE loss)
│                 │  BERTopic (19 topics, 0.96 diversity)
└────────┬────────┘
         │ model weights → HuggingFace Hub
         ▼
┌─────────────────┐
│  Stage 3        │  sentence-transformers embeddings (all-MiniLM-L6-v2)
│  RAG Pipeline   │  FAISS + ChromaDB dual indexing
│                 │  cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
└────────┬────────┘
         │ 18,188 vectors indexed
         ▼
┌─────────────────┐
│  Stage 4        │  Groq API (llama-3.1-8b-instant)
│  LLM Insights   │  Cluster summaries, trend detection, weekly reports
└────────┬────────┘
         │ structured JSON + markdown reports
         ▼
┌─────────────────┐
│  Stage 5        │  MRR, Hit Rate @K, latency benchmarks
│  Evaluation     │  Consolidated model evaluation report
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 6        │  5-page Streamlit dashboard
│  Dashboard      │  Deployed on Streamlit Community Cloud
└─────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Core ML | PyTorch 2.3, Transformers 4.44 |
| Fine-tuning | PEFT/LoRA (1.09% trainable parameters) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Topic modelling | BERTopic + UMAP + HDBSCAN |
| Vector search | FAISS + ChromaDB |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq API — Llama 3.1 8B |
| Dashboard | Streamlit |
| Model hosting | HuggingFace Hub |
| Deployment | Streamlit Community Cloud |
| Data | Amazon Reviews 2023 (HuggingFace Datasets) |

---

## Setup

### Requirements
- Python **3.11** (3.12+ not supported due to ML package compatibility)
- 8GB+ RAM recommended
- Apple Silicon (MPS), CUDA, or CPU

### Installation

```bash
git clone https://github.com/ravindrareddy-mallireddy/customer-feedback-intelligence.git
cd customer-feedback-intelligence

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

```bash
cp .env.example .env
# Add your Groq API key (free at console.groq.com)
```

### Run the pipeline

```bash
python run_stage1.py          # Data pipeline
python run_stage2_sentiment.py # Sentiment model
python run_stage2_aspect.py   # Aspect model
python run_stage2_topics.py   # Topic model
python run_stage3_retrieval.py # RAG pipeline
python run_stage4_insights.py  # LLM insights
python run_stage5_evaluation.py # Evaluation
streamlit run dashboard/app.py  # Dashboard
```

---

## Models on HuggingFace Hub

| Model | HuggingFace Repo |
|-------|-----------------|
| Sentiment | [rr1371859/customer-feedback-sentiment](https://huggingface.co/rr1371859/customer-feedback-sentiment) |
| Aspect | [rr1371859/customer-feedback-aspect](https://huggingface.co/rr1371859/customer-feedback-aspect) |
| Topics | [rr1371859/customer-feedback-topics](https://huggingface.co/rr1371859/customer-feedback-topics) |

---

## Project Structure

```
customer-feedback-intelligence/
├── config.yaml              ← All hyperparameters
├── requirements.txt         ← Python 3.11 dependencies
├── run_stage1.py            ← Data pipeline
├── run_stage2_sentiment.py  ← Sentiment training
├── run_stage2_aspect.py     ← Aspect training
├── run_stage2_topics.py     ← Topic modelling
├── run_stage3_retrieval.py  ← RAG pipeline
├── run_stage4_insights.py   ← LLM insights
├── run_stage5_evaluation.py ← Evaluation
├── data/
│   ├── processed/           ← train/val/test parquet
│   └── evaluation/          ← metrics, reports, summaries
├── src/
│   ├── data/                ← loader, preprocessor, dataloader
│   ├── models/              ← sentiment, aspect, topics
│   ├── retrieval/           ← embeddings, search, reranker
│   ├── insights/            ← summariser, trends, reports
│   └── evaluation/          ← metrics
└── dashboard/
    ├── app.py               ← Streamlit entry point
    └── pages/               ← overview, search, analyze, topics, insights
```

---

## Key Design Decisions

**Why LoRA?** Reduces trainable parameters from 67M to 740K (1.09%) while achieving 89.91% accuracy — within 1% of full fine-tuning benchmarks. Production-relevant efficiency gain.

**Why weak supervision for aspects?** No labeled aspect data available. Keyword-based pseudo-labeling is a legitimate technique used at scale. Manual annotation of ~500 reviews would push F1 from 0.35 to 0.60+.

**Why dual vector store (FAISS + ChromaDB)?** FAISS for raw speed, ChromaDB for filtered search with metadata. Demonstrates awareness of production trade-offs.

**Why Groq?** Free tier with 14,400 requests/day. Llama 3.1 8B gives GPT-3.5 quality at zero cost for structured JSON generation.

---

## Author

**Ravindra Reddy Mallireddy**
MSc Applied AI and Data Science — Southampton Solent University (Distinction, 79.6%)
[GitHub](https://github.com/ravindrareddy-mallireddy) | [LinkedIn](https://linkedin.com/in/ravindrareddy-mallireddy)
