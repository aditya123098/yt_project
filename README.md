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

# 🎬 YouTube Q&A RAG

A **Retrieval-Augmented Generation** application that lets you ask intelligent questions about any YouTube video. Built with LangChain, Streamlit, FAISS, and HuggingFace Transformers.

## ✨ Features

- **🎥 YouTube Transcript Extraction** — Automatically extracts transcripts from any YouTube video (no API key needed for transcripts)
- **✂️ Smart Text Chunking** — Splits transcripts into overlapping chunks using `RecursiveCharacterTextSplitter` for optimal context preservation
- **🧠 Semantic Embeddings** — Generates 384-dimensional embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- **📦 FAISS Vector Search** — Lightning-fast similarity search across transcript segments
- **🤖 LLM-Powered Answers** — Uses `Mistral-7B-Instruct` via HuggingFace Inference API for contextual answer generation
- **💬 Chat Interface** — Interactive conversational UI with chat history and source citations
- **🎨 Premium Dark Theme** — Glassmorphism design with gradient accents and micro-animations

## 🏗️ Architecture

```
YouTube URL → Transcript Extraction → Text Chunking → Embedding Generation
                                                           ↓
User Question → Query Embedding → FAISS Similarity Search → Top-K Chunks
                                                           ↓
                                      RAG Prompt + LLM → Answer + Sources
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- A free [HuggingFace API Token](https://huggingface.co/settings/tokens)

### Local Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd yt-project

# Install dependencies
pip install -r requirements.txt

# Set up your API token (Option A: Environment variable)
cp .env.example .env
# Edit .env and add your HuggingFace token

# Run the app
streamlit run app.py
```

### Using the App

1. **Enter your HuggingFace API token** in the sidebar (or set it as an environment variable)
2. **Paste a YouTube URL** — any video with available transcripts
3. **Click "Process Video"** — watch as the pipeline extracts, chunks, and indexes the transcript
4. **Ask questions!** — type any question about the video content

## 📁 Project Structure

```
├── app.py                # Main Streamlit application
├── rag_pipeline.py       # Core RAG pipeline (transcript, chunking, FAISS, LLM)
├── ui_components.py      # Premium dark theme CSS and reusable UI components
├── config.py             # Configuration constants
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment variables
└── README.md             # This file
```

## ⚙️ Configuration

All configuration is centralized in `config.py`:

| Parameter | Default | Description |
|:---|:---|:---|
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence transformer model for embeddings |
| `LLM_MODEL_NAME` | `Mistral-7B-Instruct-v0.3` | LLM for answer generation |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_K` | `4` | Number of chunks to retrieve |

## 🛠️ Tech Stack

| Component | Technology |
|:---|:---|
| Frontend | Streamlit |
| Orchestration | LangChain |
| Embeddings | sentence-transformers |
| Vector Store | FAISS (CPU) |
| LLM Inference | HuggingFace Inference API |
| Transcript | youtube-transcript-api |

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
