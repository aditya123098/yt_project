"""
Core RAG Pipeline for YouTube Q&A Application — powered by LangGraph.

Architecture:
  - Two LangGraph StateGraphs orchestrate the pipeline:
      1. Ingestion Graph:  URL → Transcript → Chunks → FAISS index
      2. Q&A Graph:       Question → Retrieve → Generate → Answer

  - Each pipeline step is an isolated node with typed state (TypedDict).
  - Conditional edges handle error short-circuiting explicitly.
  - The public RAGPipeline class wraps both graphs, keeping the same
    interface expected by app.py (process_video / ask).
"""

import re
import time
import logging
from typing import Optional, Callable, Any

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
from huggingface_hub import InferenceClient

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

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


# ─── Typed State Definitions ───────────────────────────────────────────────────


class IngestionState(TypedDict):
    """State that flows through the video ingestion graph."""

    # ── Inputs ──
    youtube_url: str
    progress_callback: Optional[Callable]

    # ── Intermediate data ──
    transcript_text: str
    video_id: str
    segments: list
    chunks: list

    # ── Output ──
    vector_store: Optional[Any]  # FAISS instance

    # ── Control ──
    success: bool
    error: Optional[str]
    stats: dict


class QAState(TypedDict):
    """State that flows through the Q&A graph."""

    # ── Inputs ──
    question: str
    hf_token: str
    vector_store: Any  # FAISS instance

    # ── Intermediate ──
    relevant_chunks: list

    # ── Output ──
    answer: str
    sources: list

    # ── Control ──
    success: bool
    error: Optional[str]


# ─── Shared Utility: Embedding Model ──────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load and cache the sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ─── Utility: Extract Video ID ─────────────────────────────────────────────────


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


# ═══════════════════════════════════════════════════════════════════════════════
# INGESTION GRAPH  ─  Nodes
# ═══════════════════════════════════════════════════════════════════════════════


def node_extract_transcript(state: IngestionState) -> IngestionState:
    """
    Node 1: Fetch the YouTube transcript and clean the text.

    Uses youtube-transcript-api (v1.2+ instance-based API) with exponential
    backoff retry on transient errors.
    """
    youtube_url = state["youtube_url"]
    callback = state.get("progress_callback")
    max_retries = 3

    if callback:
        callback(1, 4, "📝 Extracting transcript...")

    video_id = extract_video_id(youtube_url)
    if not video_id:
        return {
            **state,
            "success": False,
            "error": "Invalid YouTube URL. Please provide a valid YouTube video link.",
            "transcript_text": "",
            "video_id": "",
            "segments": [],
        }

    last_exception = None

    for attempt in range(max_retries):
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(video_id)

            segments_list = []
            text_parts = []
            for snippet in transcript:
                text = snippet.text if hasattr(snippet, "text") else str(snippet)
                text_parts.append(text)
                segments_list.append(
                    {
                        "text": text,
                        "start": getattr(snippet, "start", 0),
                        "duration": getattr(snippet, "duration", 0),
                    }
                )

            full_text = " ".join(text_parts)
            full_text = re.sub(r"\s+", " ", full_text).strip()
            full_text = re.sub(r"\[.*?\]", "", full_text)  # Remove [Music], etc.

            if not full_text.strip():
                return {
                    **state,
                    "success": False,
                    "error": "Transcript is empty for this video.",
                    "transcript_text": "",
                    "video_id": video_id,
                    "segments": [],
                }

            return {
                **state,
                "transcript_text": full_text,
                "video_id": video_id,
                "segments": segments_list,
                "success": True,
                "error": None,
            }

        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()

            # Permanent errors — don't retry
            if "disabled" in error_msg:
                return {
                    **state,
                    "success": False,
                    "error": "Transcripts are disabled for this video.",
                    "transcript_text": "",
                    "video_id": video_id,
                    "segments": [],
                }
            if "unavailable" in error_msg or "not exist" in error_msg:
                return {
                    **state,
                    "success": False,
                    "error": "This video is unavailable or does not exist.",
                    "transcript_text": "",
                    "video_id": video_id,
                    "segments": [],
                }
            if "no transcript" in error_msg:
                return {
                    **state,
                    "success": False,
                    "error": "No transcript found. The video may not have captions.",
                    "transcript_text": "",
                    "video_id": video_id,
                    "segments": [],
                }

            # Transient error — retry with backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                logger.warning(
                    "Transcript fetch attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt + 1,
                    max_retries,
                    e,
                    wait_time,
                )
                time.sleep(wait_time)
            else:
                logger.error("All %d transcript fetch attempts failed: %s", max_retries, e)

    # All retries exhausted
    error_msg_lower = str(last_exception).lower() if last_exception else ""
    if "blocked" in error_msg_lower:
        friendly_error = (
            "YouTube is blocking requests from this server. "
            "This is common on cloud-hosted apps. Please try again in a minute."
        )
    else:
        friendly_error = f"Error extracting transcript after {max_retries} attempts: {last_exception}"

    return {
        **state,
        "success": False,
        "error": friendly_error,
        "transcript_text": "",
        "video_id": video_id,
        "segments": [],
    }


def node_chunk_text(state: IngestionState) -> IngestionState:
    """
    Node 2: Split the transcript into overlapping chunks with metadata.

    Uses LangChain's RecursiveCharacterTextSplitter with a smart separator
    hierarchy: paragraphs → sentences → commas → words.
    """
    callback = state.get("progress_callback")
    if callback:
        callback(2, 4, "✂️ Splitting into chunks...")

    text = state["transcript_text"]
    video_id = state["video_id"]

    if not text or not text.strip():
        return {
            **state,
            "chunks": [],
            "success": False,
            "error": "Failed to create text chunks — transcript is empty.",
        }

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

    # Annotate each chunk with its index
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)

    if not chunks:
        return {
            **state,
            "chunks": [],
            "success": False,
            "error": "No chunks were produced from the transcript.",
        }

    return {**state, "chunks": chunks, "success": True, "error": None}


def node_create_index(state: IngestionState) -> IngestionState:
    """
    Node 3: Embed all chunks and build a FAISS vector store.

    Loads the cached sentence-transformer model and indexes all chunks
    for sub-millisecond similarity retrieval at query time.
    """
    callback = state.get("progress_callback")
    if callback:
        callback(3, 4, "🧠 Generating embeddings & building index...")

    chunks = state["chunks"]
    transcript_text = state["transcript_text"]
    video_id = state["video_id"]

    try:
        embeddings = load_embedding_model()
        vector_store = FAISS.from_documents(chunks, embeddings)

        if callback:
            callback(4, 4, "✅ Ready for questions!")

        stats = {
            "video_id": video_id,
            "transcript_length": len(transcript_text),
            "word_count": len(transcript_text.split()),
            "chunk_count": len(chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }

        return {
            **state,
            "vector_store": vector_store,
            "stats": stats,
            "success": True,
            "error": None,
        }

    except Exception as e:
        return {
            **state,
            "vector_store": None,
            "stats": {},
            "success": False,
            "error": f"Failed to create vector index: {e}",
        }


# ─── Ingestion Graph: Conditional Routing ──────────────────────────────────────


def route_ingestion(state: IngestionState) -> str:
    """Route to END on any failure, continue to next step on success."""
    return "continue" if state.get("success", False) else "error"


# ─── Build Ingestion Graph ──────────────────────────────────────────────────────


def build_ingestion_graph() -> StateGraph:
    """
    Compile the video ingestion StateGraph.

    Flow:
        START
          └─► extract_transcript
                ├─(success)─► chunk_text
                │               ├─(success)─► create_index ─► END
                │               └─(error)──► END
                └─(error)──► END
    """
    graph = StateGraph(IngestionState)

    # Register nodes
    graph.add_node("extract_transcript", node_extract_transcript)
    graph.add_node("chunk_text", node_chunk_text)
    graph.add_node("create_index", node_create_index)

    # Entry point
    graph.add_edge(START, "extract_transcript")

    # After extract_transcript: success → chunk_text, error → END
    graph.add_conditional_edges(
        "extract_transcript",
        route_ingestion,
        {"continue": "chunk_text", "error": END},
    )

    # After chunk_text: success → create_index, error → END
    graph.add_conditional_edges(
        "chunk_text",
        route_ingestion,
        {"continue": "create_index", "error": END},
    )

    # create_index always goes to END
    graph.add_edge("create_index", END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# Q&A GRAPH  ─  Nodes
# ═══════════════════════════════════════════════════════════════════════════════


def node_validate_input(state: QAState) -> QAState:
    """
    Node 1: Validate that the pipeline is ready and the question is non-empty.

    Short-circuits with an informative error if either check fails.
    """
    question = state.get("question", "").strip()
    vector_store = state.get("vector_store")
    hf_token = state.get("hf_token", "")

    if not vector_store:
        return {
            **state,
            "success": False,
            "error": "No video has been processed yet. Please process a video first.",
            "answer": "",
            "sources": [],
            "relevant_chunks": [],
        }

    if not question:
        return {
            **state,
            "success": False,
            "error": "Please enter a question.",
            "answer": "",
            "sources": [],
            "relevant_chunks": [],
        }

    if not hf_token:
        return {
            **state,
            "success": False,
            "error": "HuggingFace API token is required. Please enter it in the sidebar.",
            "answer": "",
            "sources": [],
            "relevant_chunks": [],
        }

    return {**state, "success": True, "error": None}


def node_retrieve_chunks(state: QAState) -> QAState:
    """
    Node 2: Retrieve the most relevant transcript chunks for the question.

    Performs cosine-similarity search against the FAISS index and returns
    the top-K (document, score) tuples.
    """
    question = state["question"]
    vector_store = state["vector_store"]

    try:
        results = vector_store.similarity_search_with_score(question, k=RETRIEVAL_K)
        return {**state, "relevant_chunks": results, "success": True, "error": None}
    except Exception as e:
        return {
            **state,
            "relevant_chunks": [],
            "success": False,
            "error": f"Retrieval error: {e}",
        }


def node_generate_answer(state: QAState) -> QAState:
    """
    Node 3: Generate a grounded answer using the HuggingFace Inference API.

    Builds the prompt from retrieved chunks, calls the LLM via chat_completion,
    and constructs the sources list for citation display.
    """
    question = state["question"]
    hf_token = state["hf_token"]
    context_chunks = state["relevant_chunks"]

    # Build context string and source metadata
    context_parts = []
    sources = []
    for i, (doc, score) in enumerate(context_chunks):
        chunk_text_content = doc.page_content.strip()
        context_parts.append(f"[Segment {i + 1}]: {chunk_text_content}")
        sources.append(
            {
                "content": chunk_text_content,
                "chunk_index": doc.metadata.get("chunk_index", i),
                "similarity_score": round(float(1 / (1 + score)), 4),
            }
        )

    context = "\n\n".join(context_parts)
    user_message = USER_PROMPT_TEMPLATE.format(context=context, question=question)

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

        answer = response.choices[0].message.content.strip()

        return {
            **state,
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
            **state,
            "answer": "",
            "sources": [],
            "success": False,
            "error": f"LLM Error: {error_msg}",
        }


# ─── Q&A Graph: Conditional Routing ────────────────────────────────────────────


def route_qa(state: QAState) -> str:
    """Route to END on failure, continue to next step on success."""
    return "continue" if state.get("success", False) else "error"


# ─── Build Q&A Graph ────────────────────────────────────────────────────────────


def build_qa_graph() -> StateGraph:
    """
    Compile the Q&A StateGraph.

    Flow:
        START
          └─► validate_input
                ├─(valid)──► retrieve_chunks ─► generate_answer ─► END
                └─(invalid)─► END
    """
    graph = StateGraph(QAState)

    # Register nodes
    graph.add_node("validate_input", node_validate_input)
    graph.add_node("retrieve_chunks", node_retrieve_chunks)
    graph.add_node("generate_answer", node_generate_answer)

    # Entry point
    graph.add_edge(START, "validate_input")

    # After validate_input: valid → retrieve_chunks, invalid → END
    graph.add_conditional_edges(
        "validate_input",
        route_qa,
        {"continue": "retrieve_chunks", "error": END},
    )

    # retrieve_chunks: success → generate_answer, error → END
    graph.add_conditional_edges(
        "retrieve_chunks",
        route_qa,
        {"continue": "generate_answer", "error": END},
    )

    # generate_answer always goes to END
    graph.add_edge("generate_answer", END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# Public RAGPipeline — wraps both LangGraph graphs
# ═══════════════════════════════════════════════════════════════════════════════


class RAGPipeline:
    """
    Public interface for the YouTube Q&A RAG pipeline.

    Internally orchestrated by two LangGraph StateGraphs:
      - ingestion_graph: processes a YouTube video end-to-end
      - qa_graph:        answers questions against the indexed video

    The public API (process_video / ask) is identical to the pre-LangGraph
    version, so app.py requires no changes.
    """

    def __init__(self):
        self.vector_store = None
        self.video_id = None
        self.is_ready = False

        # Compile both graphs once at construction time
        self._ingestion_graph = build_ingestion_graph()
        self._qa_graph = build_qa_graph()

    def process_video(self, youtube_url: str, progress_callback=None) -> dict:
        """
        Process a YouTube video through the full ingestion graph.

        Args:
            youtube_url:       The YouTube video URL to process.
            progress_callback: Optional callable(step, total, message).

        Returns:
            dict — { success: bool, error: str | None, stats: dict }
        """
        # Reset pipeline state
        self.vector_store = None
        self.video_id = None
        self.is_ready = False

        # Initial state for the ingestion graph
        initial_state: IngestionState = {
            "youtube_url": youtube_url,
            "progress_callback": progress_callback,
            "transcript_text": "",
            "video_id": "",
            "segments": [],
            "chunks": [],
            "vector_store": None,
            "success": True,
            "error": None,
            "stats": {},
        }

        # Run the ingestion graph
        final_state = self._ingestion_graph.invoke(initial_state)

        if final_state["success"]:
            self.vector_store = final_state["vector_store"]
            self.video_id = final_state["video_id"]
            self.is_ready = True
            return {"success": True, "error": None, "stats": final_state["stats"]}
        else:
            return {
                "success": False,
                "error": final_state.get("error", "An unknown error occurred."),
                "stats": {},
            }

    def ask(self, question: str, hf_token: str) -> dict:
        """
        Ask a question about the currently processed video.

        Args:
            question:  The user's question string.
            hf_token:  HuggingFace Inference API token.

        Returns:
            dict — { success: bool, answer: str, sources: list, error: str | None }
        """
        # Initial state for the Q&A graph
        initial_state: QAState = {
            "question": question,
            "hf_token": hf_token,
            "vector_store": self.vector_store,
            "relevant_chunks": [],
            "answer": "",
            "sources": [],
            "success": True,
            "error": None,
        }

        # Run the Q&A graph
        final_state = self._qa_graph.invoke(initial_state)

        return {
            "success": final_state["success"],
            "answer": final_state.get("answer", ""),
            "sources": final_state.get("sources", []),
            "error": final_state.get("error"),
        }
