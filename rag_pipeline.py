"""
Core RAG Pipeline for YouTube Q&A Application.

Handles: transcript extraction, text chunking, embedding generation,
FAISS vector store creation, and LLM-based answer generation.
"""

import re
import logging
from typing import Optional

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
from huggingface_hub import InferenceClient

from config import (
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_K,
    LLM_MODEL_NAME,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_REPETITION_PENALTY,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


# ─── YouTube Transcript Extraction ─────────────────────────────────────────────


def extract_video_id(url: str) -> Optional[str]:
    """Extract the video ID from various YouTube URL formats."""
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def extract_transcript(youtube_url: str) -> dict:
    """
    Extract transcript from a YouTube video using youtube-transcript-api v1.2+.

    Uses the new instance-based API (static methods were removed in v1.2).

    Returns:
        dict with keys: 'text', 'video_id', 'segments', 'success', 'error'
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return {
            "text": "",
            "video_id": None,
            "segments": [],
            "success": False,
            "error": "Invalid YouTube URL. Please provide a valid YouTube video link.",
        }

    try:
        # v1.2+: Use instance-based API
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)

        # Build full text from transcript snippets
        # In v1.2+, transcript is a FetchedTranscript object that is iterable
        segments_list = []
        text_parts = []
        for snippet in transcript:
            # Each snippet has .text, .start, .duration attributes
            text = snippet.text if hasattr(snippet, 'text') else str(snippet)
            text_parts.append(text)
            segments_list.append({
                "text": text,
                "start": getattr(snippet, 'start', 0),
                "duration": getattr(snippet, 'duration', 0),
            })

        full_text = " ".join(text_parts)

        # Clean up the text
        full_text = re.sub(r"\s+", " ", full_text).strip()
        full_text = re.sub(r"\[.*?\]", "", full_text)  # Remove [Music], [Applause], etc.

        if not full_text.strip():
            return {
                "text": "",
                "video_id": video_id,
                "segments": [],
                "success": False,
                "error": "Transcript is empty for this video.",
            }

        return {
            "text": full_text,
            "video_id": video_id,
            "segments": segments_list,
            "success": True,
            "error": None,
        }

    except Exception as e:
        error_msg = str(e).lower()
        if "disabled" in error_msg:
            friendly_error = "Transcripts are disabled for this video."
        elif "unavailable" in error_msg or "not exist" in error_msg:
            friendly_error = "This video is unavailable or does not exist."
        elif "no transcript" in error_msg:
            friendly_error = "No transcript found for this video. It may not have captions."
        elif "blocked" in error_msg:
            friendly_error = "Request was blocked by YouTube. Try again in a few moments."
        else:
            friendly_error = f"Error extracting transcript: {str(e)}"

        return {
            "text": "",
            "video_id": video_id,
            "segments": [],
            "success": False,
            "error": friendly_error,
        }


# ─── Text Chunking ─────────────────────────────────────────────────────────────


def chunk_text(text: str, video_id: str = "") -> list:
    """
    Split transcript text into overlapping chunks with metadata.

    Returns:
        List of LangChain Document objects with metadata.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
    )

    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": f"youtube_{video_id}", "video_id": video_id}],
    )

    # Add chunk index metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks


# ─── Embedding & FAISS Vector Store ─────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load and cache the sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(chunks: list) -> FAISS:
    """
    Create a FAISS vector store from document chunks.

    Returns:
        FAISS vector store instance.
    """
    embeddings = load_embedding_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def get_relevant_chunks(query: str, vector_store: FAISS, k: int = RETRIEVAL_K) -> list:
    """
    Retrieve the most relevant chunks for a query using similarity search.

    Returns:
        List of (Document, score) tuples, sorted by relevance.
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    return results


# ─── LLM Answer Generation ─────────────────────────────────────────────────────


def generate_answer(query: str, context_chunks: list, hf_token: str) -> dict:
    """
    Generate an answer using the HuggingFace Inference API.

    Args:
        query: The user's question.
        context_chunks: List of (Document, score) tuples from FAISS retrieval.
        hf_token: HuggingFace API access token.

    Returns:
        dict with keys: 'answer', 'sources', 'success', 'error'
    """
    if not hf_token:
        return {
            "answer": "",
            "sources": [],
            "success": False,
            "error": "HuggingFace API token is required. Please enter it in the sidebar.",
        }

    # Build context string from retrieved chunks
    context_parts = []
    sources = []
    for i, (doc, score) in enumerate(context_chunks):
        chunk_text_content = doc.page_content.strip()
        context_parts.append(f"[Segment {i + 1}]: {chunk_text_content}")
        sources.append(
            {
                "content": chunk_text_content,
                "chunk_index": doc.metadata.get("chunk_index", i),
                "similarity_score": round(float(1 / (1 + score)), 4),  # Convert distance to similarity
            }
        )

    context = "\n\n".join(context_parts)

    # Format the user message
    user_message = USER_PROMPT_TEMPLATE.format(context=context, question=query)

    try:
        client = InferenceClient(token=hf_token)
        response = client.chat_completion(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
        )

        # Extract the answer from the chat completion response
        answer = response.choices[0].message.content.strip()

        return {
            "answer": answer,
            "sources": sources,
            "success": True,
            "error": None,
        }

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            error_msg = "Invalid HuggingFace API token. Please check your token and try again."
        elif "429" in error_msg or "rate" in error_msg.lower():
            error_msg = "Rate limit exceeded. Please wait a moment and try again."
        elif "503" in error_msg or "loading" in error_msg.lower():
            error_msg = "The model is currently loading. Please wait 30-60 seconds and try again."

        return {
            "answer": "",
            "sources": [],
            "success": False,
            "error": f"LLM Error: {error_msg}",
        }


# ─── RAG Pipeline Orchestrator ──────────────────────────────────────────────────


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline:
    YouTube URL → Transcript → Chunks → Embeddings → FAISS → Ready for Q&A.
    """

    def __init__(self):
        self.transcript_data = None
        self.chunks = []
        self.vector_store = None
        self.is_ready = False
        self.video_id = None

    def process_video(self, youtube_url: str, progress_callback=None) -> dict:
        """
        Process a YouTube video through the full pipeline.

        Args:
            youtube_url: The YouTube video URL.
            progress_callback: Optional callable(step, total, message) for progress updates.

        Returns:
            dict with processing results and stats.
        """

        def update_progress(step, total, message):
            if progress_callback:
                progress_callback(step, total, message)

        # Step 1: Extract transcript
        update_progress(1, 4, "📝 Extracting transcript...")
        self.transcript_data = extract_transcript(youtube_url)

        if not self.transcript_data["success"]:
            return {
                "success": False,
                "error": self.transcript_data["error"],
                "stats": {},
            }

        self.video_id = self.transcript_data["video_id"]
        transcript_text = self.transcript_data["text"]

        # Step 2: Chunk text
        update_progress(2, 4, "✂️ Splitting into chunks...")
        self.chunks = chunk_text(transcript_text, self.video_id)

        if not self.chunks:
            return {
                "success": False,
                "error": "Failed to create text chunks from transcript.",
                "stats": {},
            }

        # Step 3: Create embeddings and FAISS index
        update_progress(3, 4, "🧠 Generating embeddings & building index...")
        self.vector_store = create_vector_store(self.chunks)

        # Step 4: Done
        update_progress(4, 4, "✅ Ready for questions!")
        self.is_ready = True

        stats = {
            "video_id": self.video_id,
            "transcript_length": len(transcript_text),
            "word_count": len(transcript_text.split()),
            "chunk_count": len(self.chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }

        return {"success": True, "error": None, "stats": stats}

    def ask(self, question: str, hf_token: str) -> dict:
        """
        Ask a question about the processed video.

        Returns:
            dict with answer, sources, and metadata.
        """
        if not self.is_ready or not self.vector_store:
            return {
                "answer": "",
                "sources": [],
                "success": False,
                "error": "No video has been processed yet. Please process a video first.",
            }

        if not question or not question.strip():
            return {
                "answer": "",
                "sources": [],
                "success": False,
                "error": "Please enter a question.",
            }

        # Retrieve relevant chunks
        relevant_chunks = get_relevant_chunks(question, self.vector_store)

        # Generate answer
        result = generate_answer(question, relevant_chunks, hf_token)

        return result
