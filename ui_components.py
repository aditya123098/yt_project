"""
UI Components and Styling for the YouTube Q&A RAG Application.

Provides premium dark-themed CSS injection and reusable Streamlit UI components
with glassmorphism, gradients, and micro-animations.
"""

import streamlit as st


def inject_custom_css():
    """Inject premium dark-themed CSS with glassmorphism and animations."""
    st.markdown(
        """
        <style>
        /* ─── Google Fonts ──────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

        /* ─── Root Variables ────────────────────────────────────────── */
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: rgba(22, 22, 35, 0.8);
            --bg-card-hover: rgba(30, 30, 48, 0.9);
            --border-color: rgba(255, 255, 255, 0.06);
            --border-glow: rgba(139, 92, 246, 0.3);
            --text-primary: #f0f0f5;
            --text-secondary: #a0a0b8;
            --text-muted: #6b6b80;
            --accent-purple: #8b5cf6;
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --accent-pink: #ec4899;
            --accent-green: #10b981;
            --gradient-primary: linear-gradient(135deg, #8b5cf6, #3b82f6, #06b6d4);
            --gradient-warm: linear-gradient(135deg, #ec4899, #8b5cf6);
            --gradient-success: linear-gradient(135deg, #10b981, #06b6d4);
            --shadow-lg: 0 20px 60px rgba(0, 0, 0, 0.4);
            --shadow-glow: 0 0 30px rgba(139, 92, 246, 0.15);
            --radius-lg: 16px;
            --radius-md: 12px;
            --radius-sm: 8px;
        }

        /* ─── Base Styles ───────────────────────────────────────────── */
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        /* ─── Hero Header ───────────────────────────────────────────── */
        .hero-header {
            text-align: center;
            padding: 2.5rem 1.5rem 2rem;
            margin-bottom: 1.5rem;
            background: linear-gradient(180deg, rgba(139, 92, 246, 0.08) 0%, transparent 100%);
            border-bottom: 1px solid var(--border-color);
        }

        .hero-icon {
            font-size: 3.2rem;
            margin-bottom: 0.5rem;
            display: inline-block;
            animation: float 3s ease-in-out infinite;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-8px); }
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #8b5cf6, #3b82f6, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0.3rem 0;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: var(--text-secondary);
            font-weight: 400;
            margin-top: 0.3rem;
        }

        /* ─── Glass Card ────────────────────────────────────────────── */
        .glass-card {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .glass-card:hover {
            background: var(--bg-card-hover);
            border-color: var(--border-glow);
            box-shadow: var(--shadow-glow);
            transform: translateY(-2px);
        }

        /* ─── Metric Cards ──────────────────────────────────────────── */
        .metric-row {
            display: flex;
            gap: 0.8rem;
            margin: 1rem 0;
        }

        .metric-card {
            flex: 1;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 1rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            border-color: var(--border-glow);
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.1);
        }

        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .metric-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 0.25rem;
        }

        /* ─── Status Badge ──────────────────────────────────────────── */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.9rem;
            border-radius: 50px;
            font-size: 0.8rem;
            font-weight: 500;
        }

        .status-ready {
            background: rgba(16, 185, 129, 0.12);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.25);
        }

        .status-processing {
            background: rgba(139, 92, 246, 0.12);
            color: #8b5cf6;
            border: 1px solid rgba(139, 92, 246, 0.25);
            animation: pulse-badge 2s ease-in-out infinite;
        }

        .status-idle {
            background: rgba(107, 107, 128, 0.12);
            color: #6b6b80;
            border: 1px solid rgba(107, 107, 128, 0.25);
        }

        @keyframes pulse-badge {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }

        /* ─── Source Card ───────────────────────────────────────────── */
        .source-card {
            background: rgba(22, 22, 35, 0.6);
            border: 1px solid var(--border-color);
            border-left: 3px solid var(--accent-purple);
            border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
            padding: 1rem 1.2rem;
            margin: 0.6rem 0;
            font-size: 0.88rem;
            line-height: 1.6;
            color: var(--text-secondary);
            transition: all 0.25s ease;
        }

        .source-card:hover {
            background: rgba(30, 30, 48, 0.7);
            border-left-color: var(--accent-cyan);
        }

        .source-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .source-label {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--accent-purple);
        }

        .similarity-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.15rem 0.55rem;
            border-radius: 50px;
            font-size: 0.7rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        .sim-high {
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
        }

        .sim-medium {
            background: rgba(245, 158, 11, 0.15);
            color: #f59e0b;
        }

        .sim-low {
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
        }

        /* ─── Sample Question Buttons ───────────────────────────────── */
        .sample-q-container {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 1rem 0;
        }

        /* ─── Processing Steps ──────────────────────────────────────── */
        .step-indicator {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.6rem 0;
            font-size: 0.9rem;
        }

        .step-done {
            color: var(--accent-green);
        }

        .step-active {
            color: var(--accent-purple);
            animation: pulse-badge 1.5s ease-in-out infinite;
        }

        .step-pending {
            color: var(--text-muted);
        }

        /* ─── Chat Message Styling ──────────────────────────────────── */
        .stChatMessage {
            border-radius: var(--radius-md) !important;
            border: 1px solid var(--border-color) !important;
            margin-bottom: 0.8rem !important;
        }

        /* ─── Sidebar Styling ───────────────────────────────────────── */
        section[data-testid="stSidebar"] {
            border-right: 1px solid var(--border-color) !important;
        }

        .sidebar-section {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 1.2rem;
            margin-bottom: 1rem;
        }

        .sidebar-section-title {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 0.8rem;
        }

        /* ─── Custom Button Styling ─────────────────────────────────── */
        .stButton > button {
            border-radius: var(--radius-sm) !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }

        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.25) !important;
        }

        /* ─── Input Styling ─────────────────────────────────────────── */
        .stTextInput > div > div > input {
            border-radius: var(--radius-sm) !important;
            border: 1px solid var(--border-color) !important;
            transition: border-color 0.3s ease !important;
        }

        .stTextInput > div > div > input:focus {
            border-color: var(--accent-purple) !important;
            box-shadow: 0 0 0 1px rgba(139, 92, 246, 0.3) !important;
        }

        /* ─── Expander Styling ──────────────────────────────────────── */
        .streamlit-expanderHeader {
            border-radius: var(--radius-sm) !important;
            font-weight: 500 !important;
        }

        /* ─── Divider ───────────────────────────────────────────────── */
        .custom-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border-color), transparent);
            margin: 1.5rem 0;
        }

        /* ─── Welcome Message ───────────────────────────────────────── */
        .welcome-container {
            text-align: center;
            padding: 3rem 2rem;
        }

        .welcome-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }

        .welcome-text {
            font-size: 1.1rem;
            color: var(--text-muted);
            max-width: 450px;
            margin: 0 auto;
            line-height: 1.7;
        }

        /* ─── Scrollbar ─────────────────────────────────────────────── */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─── Reusable UI Components ─────────────────────────────────────────────────────


def render_hero_header():
    """Render the hero header with animated icon and gradient title."""
    st.markdown(
        """
        <div class="hero-header">
            <div class="hero-icon">🎬</div>
            <div class="hero-title">YouTube Q&A RAG</div>
            <div class="hero-subtitle">Ask intelligent questions about any YouTube video — powered by AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(status: str):
    """Render a colored status badge."""
    if status == "ready":
        st.markdown(
            '<span class="status-badge status-ready">● Ready</span>',
            unsafe_allow_html=True,
        )
    elif status == "processing":
        st.markdown(
            '<span class="status-badge status-processing">◉ Processing</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge status-idle">○ Idle</span>',
            unsafe_allow_html=True,
        )


def render_metric_cards(stats: dict):
    """Render a row of metric cards with stats."""
    word_count = stats.get("word_count", 0)
    chunk_count = stats.get("chunk_count", 0)
    transcript_length = stats.get("transcript_length", 0)

    # Format large numbers
    def fmt(n):
        if n >= 1000:
            return f"{n / 1000:.1f}k"
        return str(n)

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value">{fmt(word_count)}</div>
                <div class="metric-label">Words</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{chunk_count}</div>
                <div class="metric-label">Chunks</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{fmt(transcript_length)}</div>
                <div class="metric-label">Characters</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_card(source: dict, index: int):
    """Render a source reference card with similarity score badge."""
    score = source.get("similarity_score", 0)
    content = source.get("content", "")
    chunk_idx = source.get("chunk_index", index)

    # Determine score class
    if score >= 0.7:
        sim_class = "sim-high"
        sim_icon = "▲"
    elif score >= 0.4:
        sim_class = "sim-medium"
        sim_icon = "●"
    else:
        sim_class = "sim-low"
        sim_icon = "▼"

    # Truncate content for display
    display_content = content[:300] + "..." if len(content) > 300 else content

    st.markdown(
        f"""
        <div class="source-card">
            <div class="source-header">
                <span class="source-label">📄 Segment {chunk_idx + 1}</span>
                <span class="similarity-badge {sim_class}">{sim_icon} {score:.0%}</span>
            </div>
            <div>{display_content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome_message():
    """Render the welcome message when no video is processed yet."""
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-icon">🎥</div>
            <div class="welcome-text">
                Paste a YouTube URL in the sidebar and click <strong>Process Video</strong> to get started.
                Then ask any question about the video content!
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_divider():
    """Render a subtle gradient divider."""
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
