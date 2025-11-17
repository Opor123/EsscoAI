# EsscoAI

## Overview
EsscoAI is the **central AI assistant** for Dad’s company.  
Phase 1 focuses on a **specialty chatbot** trained on curated company Q&A data, designed to provide fast and accurate answers for staff and customers.  
Future phases will expand into product, pricing, and customer support integrations.

---

## Features
- **Specialty Knowledge Base**  
  Trained on curated Q&A pairs (`training_ready.jsonl`) for domain-specific, reliable answers.

- **Smart Conversations**  
  Uses embeddings + FAISS vector search for high-confidence matching.

- **Natural Language Understanding**  
  Handles phrasing variations with confidence scoring to avoid irrelevant answers.

- **Efficient & Fast**  
  Embedding search for instant responses with lightweight architecture.

- **Extensible Architecture**  
  Easy to expand with new Q&A pairs and future LLM integration.

---

## Project Structure
```bash
EsscoAI/
├── Data/
│ ├── feedback.jsonl 
│ ├── training_ready.jsonl 
│ └── essco_ai.db
├── AI/
│ ├── feedback.py  
│ ├── Model.py
│ ├── jsonToSQL.py
│ ├── db.py
│ ├── llm_service.py
├── WebDesign/
│ ├──API/
│ │ ├──API.py
│ ├── static/
│ │ ├── main.html
│ │ ├── script.js
│ │ ├── style.css
├── main.py
└── README.md
```
---