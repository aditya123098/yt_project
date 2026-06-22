"""
YouTube Q&A RAG Application — Main Streamlit App.

A Retrieval-Augmented Generation application that lets users ask questions
about YouTube video content using transcript-based semantic search and LLM generation.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from config import (
    APP_TITLE,
    APP_ICON,
    SAMPLE_QUESTIONS,
    SAMPLE_VIDEOS,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_K,
)
from rag_pipeline import RAGPipeline
from ui_components import (
    inject_custom_css,
    render_hero_header,
    render_status_badge,
    render_metric_cards,
    render_source_card,
    render_welcome_message,
    render_divider,
)

# Load environment variables
load_dotenv()

# ─── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
inject_custom_css()


# ─── Session State Initialization ──────────────────────────────────────────────

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "pipeline": RAGPipeline(),
        "chat_history": [],
        "video_processed": False,
        "video_stats": {},
        "processing": False,
        "hf_token": os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ─── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with configuration and video processing controls."""
    with st.sidebar:
        # Logo and title
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0 0.5rem;">
                <span style="font-size: 2rem;">🎬</span>
                <div style="font-size: 1.2rem; font-weight: 700; 
                     background: linear-gradient(135deg, #8b5cf6, #3b82f6);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                     background-clip: text; margin-top: 0.3rem;">
                    YouTube Q&A RAG
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_divider()

        # ── API Token Section ──
        st.markdown(
            '<div class="sidebar-section-title">🔑 API Configuration</div>',
            unsafe_allow_html=True,
        )

        hf_token = st.text_input(
            "HuggingFace API Token",
            value=st.session_state.hf_token,
            type="password",
            placeholder="hf_...",
            help="Get your free token at huggingface.co/settings/tokens",
            key="hf_token_input",
        )

        if hf_token != st.session_state.hf_token:
            st.session_state.hf_token = hf_token

        if not hf_token:
            st.warning("⚠️ Enter your HF token to enable Q&A", icon="🔑")

        render_divider()

        # ── Video Input Section ──
        st.markdown(
            '<div class="sidebar-section-title">🎥 Video Input</div>',
            unsafe_allow_html=True,
        )

        youtube_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video URL with available transcripts",
            key="youtube_url_input",
        )

        # Sample videos
        st.markdown(
            '<p style="font-size: 0.75rem; color: #6b6b80; margin-top: 0.5rem;">Try a sample video:</p>',
            unsafe_allow_html=True,
        )
        for sample in SAMPLE_VIDEOS:
            if st.button(
                f"▶ {sample['title']}",
                key=f"sample_{sample['url']}",
                use_container_width=True,
            ):
                st.session_state["youtube_url_input"] = sample["url"]
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Process button
        process_disabled = not youtube_url or st.session_state.processing
        if st.button(
            "🚀 Process Video",
            type="primary",
            use_container_width=True,
            disabled=process_disabled,
            key="process_btn",
        ):
            process_video(youtube_url)

        render_divider()

        # ── Status Section ──
        st.markdown(
            '<div class="sidebar-section-title">📊 Status</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.video_processed:
            render_status_badge("ready")
            stats = st.session_state.video_stats
            render_metric_cards(stats)

            # Video ID link
            video_id = stats.get("video_id", "")
            if video_id:
                st.markdown(
                    f'<a href="https://youtube.com/watch?v={video_id}" target="_blank" '
                    f'style="font-size: 0.8rem; color: #8b5cf6;">🔗 View on YouTube</a>',
                    unsafe_allow_html=True,
                )
        elif st.session_state.processing:
            render_status_badge("processing")
        else:
            render_status_badge("idle")
            st.caption("No video processed yet")

        render_divider()

        # ── Model Info Section ──
        st.markdown(
            '<div class="sidebar-section-title">⚙️ Pipeline Config</div>',
            unsafe_allow_html=True,
        )

        with st.expander("View Details", expanded=False):
            st.markdown(
                f"""
                **Embedding Model:**  
                `{EMBEDDING_MODEL_NAME.split('/')[-1]}`
                
                **LLM Model:**  
                `{LLM_MODEL_NAME.split('/')[-1]}`
                
                **Chunk Size:** {CHUNK_SIZE} chars  
                **Chunk Overlap:** {CHUNK_OVERLAP} chars  
                **Retrieval K:** {RETRIEVAL_K} chunks
                """,
            )

        # Clear conversation button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()


# ─── Video Processing ──────────────────────────────────────────────────────────

def process_video(youtube_url: str):
    """Process a YouTube video through the RAG pipeline."""
    st.session_state.processing = True
    st.session_state.video_processed = False
    st.session_state.chat_history = []

    pipeline = RAGPipeline()
    st.session_state.pipeline = pipeline

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    def progress_callback(step, total, message):
        progress_bar.progress(step / total)
        status_text.markdown(
            f'<p style="font-size: 0.85rem; color: #a0a0b8;">{message}</p>',
            unsafe_allow_html=True,
        )

    result = pipeline.process_video(youtube_url, progress_callback)

    if result["success"]:
        st.session_state.video_processed = True
        st.session_state.video_stats = result["stats"]
        status_text.success("✅ Video processed successfully!")
    else:
        status_text.error(f"❌ {result['error']}")

    st.session_state.processing = False
    progress_bar.empty()
    st.rerun()


# ─── Main Chat Interface ──────────────────────────────────────────────────────

def render_main_content():
    """Render the main content area with chat interface."""

    # Hero header
    render_hero_header()

    if not st.session_state.video_processed:
        render_welcome_message()

        # Quick start guide
        st.markdown("---")
        cols = st.columns(3)
        steps = [
            ("1️⃣", "Add API Token", "Get a free token from HuggingFace and paste it in the sidebar"),
            ("2️⃣", "Paste Video URL", "Enter any YouTube video URL that has transcripts available"),
            ("3️⃣", "Ask Questions", "Once processed, ask anything about the video content"),
        ]
        for col, (icon, title, desc) in zip(cols, steps):
            with col:
                st.markdown(
                    f"""
                    <div class="glass-card" style="text-align: center; min-height: 160px;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                        <div style="font-weight: 600; font-size: 1rem; margin-bottom: 0.4rem;">{title}</div>
                        <div style="font-size: 0.85rem; color: #a0a0b8;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        return

    # ── Chat History ──
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander(f"📄 View Sources ({len(message['sources'])} segments)", expanded=False):
                    for i, source in enumerate(message["sources"]):
                        render_source_card(source, i)

    # ── Sample Questions (only if no chat history) ──
    if not st.session_state.chat_history:
        st.markdown(
            '<p style="text-align: center; color: #6b6b80; font-size: 0.9rem; margin-top: 1rem;">'
            "✨ Try one of these questions or type your own below:</p>",
            unsafe_allow_html=True,
        )

        cols = st.columns(2)
        for i, question in enumerate(SAMPLE_QUESTIONS):
            with cols[i % 2]:
                if st.button(
                    f"💬 {question}",
                    key=f"sample_q_{i}",
                    use_container_width=True,
                ):
                    handle_question(question)

    # ── Chat Input ──
    if prompt := st.chat_input("Ask a question about the video...", key="chat_input"):
        handle_question(prompt)


def handle_question(question: str):
    """Handle a user question through the RAG pipeline."""
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": question})

    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching transcript & generating answer..."):
            result = st.session_state.pipeline.ask(
                question, st.session_state.hf_token
            )

        if result["success"]:
            st.markdown(result["answer"])

            # Show sources
            if result["sources"]:
                with st.expander(
                    f"📄 View Sources ({len(result['sources'])} segments)",
                    expanded=False,
                ):
                    for i, source in enumerate(result["sources"]):
                        render_source_card(source, i)

            # Save to history
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                }
            )
        else:
            error_msg = f"⚠️ {result['error']}"
            st.error(error_msg)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": error_msg, "sources": []}
            )

    st.rerun()


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def main():
    """Main application entry point."""
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
