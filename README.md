# Customer Feedback Intelligence Platform

An end-to-end NLP pipeline that fine-tunes transformer models on 18,000+ real Amazon product reviews.

Python 3.11 required.

## Results

| Model | Metric | Score |
|-------|--------|-------|
| Sentiment (DistilBERT + LoRA) | Accuracy | 89.91% |
| Sentiment (DistilBERT + LoRA) | F1 Weighted | 0.8991 |
| Aspect (DistilBERT + LoRA) | F1 Micro | 0.3503 |
| Topics (BERTopic) | Topics Found | 19 |
| Retrieval (FAISS + Reranker) | MRR | 1.0000 |
| Retrieval (FAISS + Reranker) | Latency P50 | 215ms |

## Models

- Sentiment: https://huggingface.co/rr1371859/customer-feedback-sentiment
- Aspect: https://huggingface.co/rr1371859/customer-feedback-aspect
- Topics: https://huggingface.co/rr1371859/customer-feedback-topics
