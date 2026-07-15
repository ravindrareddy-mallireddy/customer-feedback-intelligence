# Customer Feedback Intelligence Platform

An end-to-end NLP pipeline that fine-tunes transformer models on 18,000+ real Amazon product reviews, builds a semantic search engine, generates automated insights using LLMs, and presents everything in an interactive live dashboard.

**Live Demo:** https://customer-feedback-intel.streamlit.app

---

## Results

| Model | Metric | Score |
|-------|--------|-------|
| Sentiment (DistilBERT + LoRA) | Accuracy | 89.91% |
| Sentiment (DistilBERT + LoRA) | F1 Weighted | 0.8991 |
| Sentiment (DistilBERT + LoRA) | Trainable Params | 1.09% |
| Aspect (DistilBERT + LoRA) | F1 Micro | 0.3503 |
| Aspect (DistilBERT + LoRA) | F1 Macro | 0.3201 |
| Topics (BERTopic) | Topics Found | 19 |
| Topics (BERTopic) | Topic Diversity | 0.9579 |
| Retrieval (FAISS + Reranker) | MRR | 1.0000 |
| Retrieval (FAISS + Reranker) | Hit Rate at 5 | 1.0000 |
| Retrieval (FAISS + Reranker) | Latency P50 | 215ms |
| Retrieval (FAISS + Reranker) | Latency P95 | 536ms |

---

## What it does

- **Sentiment Analysis** - Classifies reviews as positive/negative using DistilBERT + LoRA (89.91% accuracy, only 1.09% trainable parameters)
- **Aspect Detection** - Identifies which product dimensions are mentioned: price, quality, delivery, packaging, usability, customer service
- **Topic Clustering** - Groups 18,188 reviews into 19 meaningful topics using BERTopic with UMAP and HDBSCAN
- **Semantic Search** - Search reviews in plain English with FAISS and ChromaDB dual indexing, cross-encoder reranking (MRR = 1.0)
- **LLM Insights** - Auto-generates cluster summaries, trend reports, and weekly insight documents using Groq API (Llama 3.1 8B)
- **Interactive Dashboard** - 5-page Streamlit app with charts, semantic search, real-time review analysis, and downloadable reports

---

## Architecture

Stage 1: Data Pipeline - Amazon Reviews download, cleaning, 80/10/10 stratified split
Stage 2: Fine-tuning - DistilBERT + LoRA for sentiment and aspect, BERTopic for topics
Stage 3: RAG Pipeline - sentence-transformers embeddings, FAISS + ChromaDB indexing, cross-encoder reranking
Stage 4: LLM Insights - Groq API cluster summaries, trend detection, weekly report generation
Stage 5: Evaluation - MRR, Hit Rate, latency benchmarks, consolidated model report
Stage 6: Dashboard - 5-page Streamlit app deployed on Streamlit Community Cloud

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Core ML | PyTorch, Transformers 4.44 |
| Fine-tuning | PEFT/LoRA (1.09% trainable parameters) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Topic modelling | BERTopic + UMAP + HDBSCAN |
| Vector search | FAISS + ChromaDB |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq API - Llama 3.1 8B Instant |
| Dashboard | Streamlit |
| Model hosting | HuggingFace Hub |
| Deployment | Streamlit Community Cloud |
| Data | Amazon Reviews 2023 (HuggingFace Datasets) |

---

## Requirements

- Python 3.11 (Python 3.12+ not supported due to ML package compatibility)
- 8GB+ RAM recommended
- Apple Silicon (MPS), CUDA, or CPU

---

## Setup

```
git clone https://github.com/ravindrareddy-mallireddy/customer-feedback-intelligence.git
cd customer-feedback-intelligence
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add GROQ_API_KEY to .env - free at console.groq.com
```

## Run Pipeline

```
python run_stage1.py
python run_stage2_sentiment.py
python run_stage2_aspect.py
python run_stage2_topics.py
python run_stage3_retrieval.py
python run_stage4_insights.py
python run_stage5_evaluation.py
streamlit run dashboard/app.py
```

---

## Models on HuggingFace Hub

| Model | Link |
|-------|------|
| Sentiment | https://huggingface.co/rr1371859/customer-feedback-sentiment |
| Aspect | https://huggingface.co/rr1371859/customer-feedback-aspect |
| Topics | https://huggingface.co/rr1371859/customer-feedback-topics |

---

## Key Design Decisions

**Why LoRA?** Reduces trainable parameters from 67M to 740K (1.09%) while achieving 89.91% accuracy - within 1% of full fine-tuning benchmarks. Production-relevant efficiency gain.

**Why weak supervision for aspects?** No labeled aspect data available. Keyword-based pseudo-labeling is a legitimate technique used at scale. Manual annotation of 500 reviews would push F1 from 0.35 to 0.60+.

**Why dual vector store (FAISS + ChromaDB)?** FAISS for raw speed, ChromaDB for filtered search with metadata. Demonstrates awareness of production trade-offs.

**Why Groq?** Free tier with 14,400 requests/day. Llama 3.1 8B gives strong quality at zero cost for structured JSON generation.

---

## Project Structure

```
customer-feedback-intelligence/
├── config.yaml                  All hyperparameters, no hardcoded values
├── requirements.txt             Python 3.11 dependencies
├── run_stage1.py                Data pipeline
├── run_stage2_sentiment.py      Sentiment training
├── run_stage2_aspect.py         Aspect training
├── run_stage2_topics.py         Topic modelling
├── run_stage3_retrieval.py      RAG pipeline
├── run_stage4_insights.py       LLM insights
├── run_stage5_evaluation.py     Evaluation
├── data/
│   ├── processed/               train/val/test parquet files
│   └── evaluation/              metrics, reports, cluster summaries
├── src/
│   ├── data/                    loader, preprocessor, dataloader, visualise
│   ├── models/                  sentiment, aspect, topics
│   ├── retrieval/               embeddings, search, reranker
│   ├── insights/                summariser, trends, reports
│   └── evaluation/              metrics
└── dashboard/
    ├── app.py                   Streamlit entry point
    └── pages/                   overview, search, analyze, topics, insights
```

---

## Author

Ravindra Reddy Mallireddy
MSc Applied AI and Data Science, Southampton Solent University (Distinction, 79.6%)
GitHub: https://github.com/ravindrareddy-mallireddy
LinkedIn: https://linkedin.com/in/ravindrareddy-mallireddy
"""
with open('README.md', 'w') as f:
    f.write(content)
print('Done')
