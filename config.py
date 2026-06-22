"""
Configuration constants for the YouTube Q&A RAG Application.
"""

# ─── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ─── LLM Model (HuggingFace Inference API) ─────────────────────────────────────
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.3
LLM_REPETITION_PENALTY = 1.1

# ─── Text Chunking ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ─── Retrieval ──────────────────────────────────────────────────────────────────
RETRIEVAL_K = 4  # Number of chunks to retrieve

# ─── RAG Prompt Template ───────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """<s>[INST] You are a helpful AI assistant that answers questions based on YouTube video transcripts.

Use ONLY the following context from the video transcript to answer the question. Follow these rules strictly:
1. Answer based ONLY on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say "I couldn't find enough information in this video to answer that question."
3. Be concise but thorough. Use bullet points for multi-part answers.
4. When relevant, mention which part of the discussion your answer comes from.

CONTEXT FROM VIDEO TRANSCRIPT:
{context}

QUESTION: {question}

ANSWER: [/INST]"""

# ─── UI Constants ──────────────────────────────────────────────────────────────
APP_TITLE = "YouTube Q&A RAG"
APP_SUBTITLE = "Ask questions about any YouTube video"
APP_ICON = "🎬"

SAMPLE_QUESTIONS = [
    "What is the main topic discussed in this video?",
    "Can you summarize the key points?",
    "What examples or case studies were mentioned?",
    "What conclusions or recommendations were given?",
]

SAMPLE_VIDEOS = [
    {
        "title": "3Blue1Brown — Neural Networks",
        "url": "https://www.youtube.com/watch?v=aircAruvnKk",
    },
    {
        "title": "Fireship — 100 Seconds of Code",
        "url": "https://www.youtube.com/watch?v=DC471a9qrU4",
    },
]
