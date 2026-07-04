---
title: YouTube Q&A RAG
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

<div align="center">

# 🎬 YouTube Q&A RAG

### *Intelligent Question-Answering over YouTube Videos — Orchestrated by LangGraph*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

A production-ready **Retrieval-Augmented Generation (RAG)** application that extracts YouTube video transcripts, builds a semantic search index, and answers user questions using open-source LLMs — all orchestrated by **LangGraph StateGraphs** through an interactive, premium-themed Streamlit interface.

[Get Started](#-quick-start) · [Architecture](#-langgraph-architecture) · [Tech Stack](#-tech-stack) · [Deploy](#-deployment)

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [LangGraph Architecture](#-langgraph-architecture)
- [How the RAG Pipeline Works](#-how-the-rag-pipeline-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Configuration](#-configuration)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🎯 Overview

**YouTube Q&A RAG** solves a simple but powerful problem: *"I watched a long YouTube video — can I just ask it questions instead of re-watching?"*

This application implements a complete **end-to-end RAG pipeline** orchestrated by **LangGraph**:

1. **Extracts** the transcript from any YouTube video (no YouTube API key required)
2. **Chunks** the transcript into semantically meaningful segments with overlap
3. **Embeds** each chunk into a 384-dimensional vector space using sentence-transformers
4. **Indexes** all vectors in a FAISS vector database for sub-millisecond retrieval
5. **Retrieves** the most relevant transcript segments for any user question
6. **Generates** a grounded, contextual answer using an open-source LLM

All pipeline steps are modelled as **LangGraph nodes** with typed state, explicit conditional routing, and clean error handling — wrapped in a **premium dark-themed Streamlit UI** with chat history, source citations, and real-time processing feedback.

---

## ✨ Key Features

### 🔀 LangGraph-Orchestrated Pipeline
- Two declarative `StateGraph` instances manage the full lifecycle
- Each processing step is an **isolated, testable node** with typed `TypedDict` state
- **Conditional edges** route around errors — no nested try/except spaghetti
- State is immutable and fully inspectable between steps
- Easy to extend with new nodes (summarization, tool-calling, re-ranking, etc.)

### 🎥 YouTube Transcript Extraction
- Automatically extracts transcripts using `youtube-transcript-api` (v1.2+)
- Supports multiple YouTube URL formats (`youtube.com/watch?v=`, `youtu.be/`, embed links)
- No Google API key required — works directly with YouTube's transcript system
- Handles edge cases: disabled transcripts, unavailable videos, empty captions
- Exponential backoff retry for transient cloud-blocking errors

### ✂️ Intelligent Text Chunking
- Uses LangChain's `RecursiveCharacterTextSplitter` for context-preserving splits
- Configurable chunk size (default: 1000 chars) and overlap (default: 200 chars)
- Smart splitting hierarchy: paragraphs → sentences → commas → words

### 🧠 Semantic Search with FAISS
- Sentence-transformer embeddings (`all-MiniLM-L6-v2`, 384 dimensions)
- L2-normalized vectors for accurate cosine similarity
- Cached embedding model — no re-loading between queries
- Retrieves top-K most relevant chunks per question (default K=4)

### 🤖 Open-Source LLM Generation
- Uses `mistralai/Mistral-7B-Instruct-v0.3` via HuggingFace Inference API
- Strict grounding prompt — answers only from transcript context
- Source citations with similarity scores shown in expandable UI panels

### 🎨 Premium Streamlit UI
- Dark glassmorphism theme with animated hero header
- Real-time progress feedback during video processing
- Chat history with per-message source citations
- Sample videos and sample questions for quick exploration

---

## 🔀 LangGraph Architecture

The core RAG pipeline is split into **two LangGraph StateGraphs**:

### Graph 1 — Ingestion Graph

Processes a YouTube URL into a searchable FAISS index.

```
START
  └─► [extract_transcript]
        │  State: youtube_url → transcript_text, video_id, segments
        │
        ├─(success)─► [chunk_text]
        │               │  State: transcript_text → chunks (LangChain Documents)
        │               │
        │               ├─(success)─► [create_index]
        │               │               │  State: chunks → vector_store (FAISS), stats
        │               │               └─► END ✅
        │               │
        │               └─(error)──► END ❌
        │
        └─(error)──► END ❌
```

### Graph 2 — Q&A Graph

Answers a question against the indexed video.

```
START
  └─► [validate_input]
        │  Checks: vector_store loaded? question non-empty? hf_token present?
        │
        ├─(valid)──► [retrieve_chunks]
        │               │  State: question → relevant_chunks (top-K FAISS results)
        │               │
        │               └─(success)─► [generate_answer]
        │                               │  State: relevant_chunks + question → answer, sources
        │                               └─► END ✅
        │
        └─(invalid)─► END ❌
```

### Typed State

Each graph uses a `TypedDict` to explicitly declare all state fields:

```python
class IngestionState(TypedDict):
    youtube_url: str
    progress_callback: Optional[Callable]
    transcript_text: str
    video_id: str
    segments: list
    chunks: list
    vector_store: Optional[FAISS]
    success: bool
    error: Optional[str]
    stats: dict

class QAState(TypedDict):
    question: str
    hf_token: str
    vector_store: FAISS
    relevant_chunks: list
    answer: str
    sources: list
    success: bool
    error: Optional[str]
```

---

## 🔧 How the RAG Pipeline Works

```
YouTube URL
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                   INGESTION GRAPH (LangGraph)                │
│                                                             │
│  extract_transcript → chunk_text → create_index             │
│   (with retry)       (1000 chars   (FAISS + all-MiniLM)     │
│                       200 overlap)                          │
└─────────────────────────────────────────────────────────────┘
     │ vector_store
     ▼
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                      Q&A GRAPH (LangGraph)                   │
│                                                             │
│  validate_input → retrieve_chunks → generate_answer         │
│  (guards)          (top-4 FAISS)    (Mistral-7B via HF API) │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
  Answer + Sources
```

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Orchestration** | [LangGraph](https://langchain-ai.github.io/langgraph/) | StateGraph-based pipeline orchestration |
| **UI** | [Streamlit](https://streamlit.io) | Interactive web interface |
| **Transcript** | [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) | YouTube caption extraction |
| **Chunking** | [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) | Recursive character splitting |
| **Embeddings** | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384-dim semantic embeddings |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) | Sub-millisecond similarity search |
| **LLM** | [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | Answer generation |
| **LLM API** | [HuggingFace Inference API](https://huggingface.co/inference-api) | Serverless LLM inference |
| **State Types** | [typing_extensions TypedDict](https://pypi.org/project/typing-extensions/) | Explicit typed state contracts |

---

## 📁 Project Structure

```
yt_project/
├── app.py                  # Streamlit entrypoint (UI, session state, routing)
├── rag_pipeline.py         # ⭐ LangGraph pipeline (two StateGraphs + RAGPipeline class)
├── config.py               # Model names, chunking params, prompt templates, UI constants
├── ui_components.py        # Premium CSS injection + reusable Streamlit components
├── requirements.txt        # Python dependencies (includes langgraph)
├── .env.example            # Environment variable template
├── .streamlit/
│   ├── config.toml         # Streamlit dark theme configuration
│   └── secrets.toml.example # Secrets template for Streamlit Cloud
└── README.md               # This file
```

### Key file: `rag_pipeline.py`

```
rag_pipeline.py
├── IngestionState (TypedDict)   — state contract for Graph 1
├── QAState (TypedDict)          — state contract for Graph 2
│
├── INGESTION GRAPH NODES
│   ├── node_extract_transcript  — fetch & clean YouTube transcript
│   ├── node_chunk_text          — split transcript into overlapping chunks
│   └── node_create_index        — embed chunks & build FAISS index
│
├── Q&A GRAPH NODES
│   ├── node_validate_input      — guard: pipeline ready? question valid? token present?
│   ├── node_retrieve_chunks     — FAISS similarity search (top-K)
│   └── node_generate_answer     — LLM answer generation via HF Inference API
│
├── build_ingestion_graph()      — compile & return ingestion StateGraph
├── build_qa_graph()             — compile & return Q&A StateGraph
│
└── RAGPipeline (class)          — public wrapper; exposes process_video() + ask()
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- A free [HuggingFace account](https://huggingface.co/join) with an API token
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/aditya123098/yt_project.git
cd yt_project
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **First run** downloads the `all-MiniLM-L6-v2` embedding model (~90 MB). It is cached for subsequent runs.

### 4. Configure Your API Token

**Option A — `.env` file (local development):**

```bash
cp .env.example .env
# Edit .env and set:
# HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

**Option B — Enter in sidebar** at runtime (no file needed).

### 5. Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. 🎉

---

## 📖 Usage Guide

### Processing a Video

1. **Enter API Token** — Paste your HuggingFace token in the sidebar (or set it in `.env`)
2. **Paste a YouTube URL** — Any video with captions enabled works
   - Try a **sample video** by clicking one of the quick-start buttons
3. **Click "🚀 Process Video"** — Watch the 4-step progress bar:
   - 📝 Extracting transcript...
   - ✂️ Splitting into chunks...
   - 🧠 Generating embeddings & building index...
   - ✅ Ready for questions!
4. **Ask Questions** — Type in the chat box or click a sample question

### Asking Questions

Questions work best when they are specific:

| ✅ Good | ❌ Vague |
|---|---|
| "What examples of backpropagation were mentioned?" | "Tell me everything" |
| "What conclusion did the presenter reach?" | "Summarize" |
| "How did they explain gradient descent?" | "What was good?" |

---

## ⚙️ Configuration

All tunable parameters live in [`config.py`](config.py):

```python
# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM (HuggingFace Inference API)
LLM_MODEL_NAME        = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_MAX_NEW_TOKENS    = 512
LLM_TEMPERATURE       = 0.3
LLM_REPETITION_PENALTY = 1.1

# Chunking
CHUNK_SIZE    = 1000   # characters per chunk
CHUNK_OVERLAP = 200    # characters of overlap between chunks

# Retrieval
RETRIEVAL_K = 4        # number of chunks to retrieve per query
```

---

## 🌐 Deployment

### Streamlit Cloud (Recommended — Free)

1. Fork this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your fork, branch `main`, and `app.py` as the main file
4. In **Settings → Secrets**, add:
   ```toml
   HUGGINGFACEHUB_API_TOKEN = "hf_your_token_here"
   ```
5. Deploy! ✅

### HuggingFace Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Streamlit** as the SDK
3. Push this repository to the Space
4. Add `HUGGINGFACEHUB_API_TOKEN` in Space settings → Variables and secrets

### Local Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t yt-qa-rag .
docker run -p 8501:8501 -e HUGGINGFACEHUB_API_TOKEN=hf_... yt-qa-rag
```

---

## 🔧 Troubleshooting

### "YouTube is blocking requests from this server"

This happens when YouTube rate-limits cloud datacenter IPs. The pipeline already implements exponential backoff (2s → 4s → 8s). If it persists:
- Wait a minute and try again
- Try a different video
- Run the app locally instead of in the cloud

### "Transcripts are disabled for this video"

The video creator has disabled auto-generated captions. Try a different video — most educational/tutorial videos have transcripts available.

### "The model is currently loading. Please wait..."

HuggingFace free tier models cold-start (~30s). Wait 30-60 seconds then retry your question.

### "Invalid HuggingFace API token"

- Ensure your token starts with `hf_`
- Check that the token has **read** permissions at huggingface.co/settings/tokens
- On Streamlit Cloud, ensure the secret is named exactly `HUGGINGFACEHUB_API_TOKEN`

### Model download is slow

The `all-MiniLM-L6-v2` model (~90 MB) is downloaded once and cached automatically. Subsequent runs use the cache.

---

## 🤝 Contributing

Contributions are welcome! The LangGraph architecture makes it easy to add new nodes:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-new-node`
3. Add a new node function to `rag_pipeline.py` and wire it into the appropriate graph
4. Submit a pull request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Built with ❤️ using LangGraph, LangChain, Streamlit, and HuggingFace

</div>
