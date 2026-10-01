import json
import os
import re
import time

import chromadb
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit as st

# ============================================================
# Safe Optional Imports (Prevents crashes if packages are missing)
# ============================================================
try:
    import requests
    from bs4 import BeautifulSoup
    from duckduckgo_search import DDGS
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False

# ============================================================
# Page Config & Metadata (Sidebar expanded like ChatGPT)
# ============================================================
st.set_page_config(
    page_title="ABES Campus AI • Intelligent University Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FOLDER = "data"
DATASET_FOLDER = "dataset"
DEFAULT_FALLBACK_MODEL = "openai/gpt-oss-20b"
ABES_DOMAIN = "abes.ac.in"
CHAT_HISTORY_FILE = "chat_history.json"


def get_question_papers_folder():
    candidates = [
        "question_papers",
        "question_paper",
        "questionpapers",
        "pyq",
        "pyqs",
        "data/question_papers",
    ]
    for folder in candidates:
        if os.path.exists(folder) and os.path.isdir(folder):
            return folder
    return "question_papers"


QUESTION_PAPERS_FOLDER = get_question_papers_folder()

# ============================================================
# Persistent Chat Storage (Saves conversations like ChatGPT)
# ============================================================
def load_all_conversations():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_all_conversations(conversations):
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ============================================================
# Royal Amethyst / Velvet Iris Theme CSS + ChatGPT Sidebar
# ============================================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        letter-spacing: -0.015em;
    }

    /* Entire App Background: Deep Obsidian & Royal Amethyst Mesh */
    .stApp, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% -10%, #1c1438 0%, #0d0f18 45%, #07080d 100%) !important;
        color: #f1f5f9 !important;
    }

    #MainMenu, footer {
        visibility: hidden !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    /* Ensure the sidebar toggle icon stays visible & clickable */
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        display: block !important;
        color: #c084fc !important;
    }

    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 6rem !important;
        max-width: 860px !important;
        margin: 0 auto !important;
    }

    /* ============================================================ */
    /* 1. CHATGPT-STYLE SIDEBAR STYLING                             */
    /* ============================================================ */
    [data-testid="stSidebar"] {
        background-color: #0b0d14 !important;
        border-right: 1px solid rgba(168, 85, 247, 0.22) !important;
        min-width: 290px !important;
        max-width: 320px !important;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 4px 14px 4px;
        border-bottom: 1px solid rgba(168, 85, 247, 0.2);
        margin-bottom: 12px;
    }
    .sidebar-brand-title {
        font-weight: 800;
        font-size: 1.15rem;
        background: linear-gradient(135deg, #ffffff 40%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .sidebar-section-title {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        color: #94a3b8;
        margin-top: 14px;
        margin-bottom: 6px;
    }

    /* Conversation Item in Sidebar */
    .convo-btn button {
        background: transparent !important;
        color: #94a3b8 !important;
        border: none !important;
        text-align: left !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 8px 10px !important;
        border-radius: 8px !important;
        display: block !important;
        width: 100% !important;
        transition: all 0.15s ease !important;
    }
    .convo-btn button:hover {
        background: rgba(168, 85, 247, 0.15) !important;
        color: #ffffff !important;
    }
    .convo-btn-active button {
        background: rgba(168, 85, 247, 0.25) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-left: 3px solid #a855f7 !important;
        border-radius: 4px !important;
    }

    .del-btn button {
        background: transparent !important;
        color: #64748b !important;
        border: none !important;
        padding: 6px !important;
        font-size: 0.85rem !important;
    }
    .del-btn button:hover {
        color: #ef4444 !important;
    }

    /* ============================================================ */
    /* 2. BRAND HERO HEADER (Centered & Fascinating)                */
    /* ============================================================ */
    .brand-hero {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(168, 85, 247, 0.35);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 24px 28px;
        margin-bottom: 2rem;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .brand-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 100px;
        padding: 5px 15px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .live-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
    }
    .brand-title {
        font-size: 2.35rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        background: linear-gradient(135deg, #ffffff 20%, #c084fc 70%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.8px;
        line-height: 1.15;
    }
    .brand-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 8px;
        line-height: 1.5;
    }
    .pill-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
        margin-top: 14px;
    }
    .pill-tag {
        background: rgba(168, 85, 247, 0.1);
        border: 1px solid rgba(168, 85, 247, 0.25);
        border-radius: 8px;
        padding: 4px 11px;
        font-size: 0.75rem;
        color: #d8b4fe;
        font-weight: 600;
    }

    /* ============================================================ */
    /* 3. CHAT MESSAGE BUBBLES                                      */
    /* ============================================================ */
    div[data-testid="stChatMessage"] {
        padding: 16px 20px !important;
        border-radius: 18px !important;
        margin-bottom: 1.2rem !important;
    }

    /* User Bubble: Deep Royal Indigo / Violet */
    div[data-testid="stChatMessage"]:has(div.user-hook) {
        background: linear-gradient(135deg, #1e1b4b 0%, #2e1065 60%, #4338ca 100%) !important;
        border: 1.5px solid rgba(168, 85, 247, 0.55) !important;
        border-radius: 20px 20px 4px 20px !important;
        box-shadow: 0 6px 24px rgba(46, 16, 101, 0.45) !important;
        margin-left: auto !important;
        max-width: 86% !important;
    }
    div[data-testid="stChatMessage"]:has(div.user-hook) * {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* Assistant Bubble: Velvet Iris / Royal Amethyst Accent */
    div[data-testid="stChatMessage"]:has(div.bot-hook) {
        background: linear-gradient(145deg, rgba(20, 21, 33, 0.95) 0%, rgba(13, 15, 24, 0.98) 100%) !important;
        border: 1px solid rgba(168, 85, 247, 0.32) !important;
        border-left: 5px solid #a855f7 !important;
        border-radius: 4px 22px 22px 22px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45), 0 0 20px rgba(168, 85, 247, 0.08) !important;
        backdrop-filter: blur(14px) !important;
        margin-right: auto !important;
        max-width: 95% !important;
    }
    div[data-testid="stChatMessage"]:has(div.bot-hook) * {
        color: #f1f5f9 !important;
    }
    div[data-testid="stChatMessage"] p {
        line-height: 1.7 !important;
        font-size: 0.96rem !important;
    }
    div[data-testid="stChatMessage"] strong {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* ============================================================ */
    /* 4. PROMPT BUTTONS & TABLES                                   */
    /* ============================================================ */
    div[data-testid="stButton"] button {
        border-radius: 14px !important;
        border: 1.5px solid rgba(168, 85, 247, 0.3) !important;
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.45) 0%, rgba(15, 23, 42, 0.65) 100%) !important;
        padding: 13px 18px !important;
        text-align: left !important;
        font-weight: 600 !important;
        font-size: 0.91rem !important;
        color: #f8fafc !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stButton"] button:hover {
        border-color: #c084fc !important;
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.25) 0%, rgba(99, 102, 241, 0.22) 100%) !important;
        transform: translateY(-2px) !important;
        color: #ffffff !important;
    }

    /* Download Buttons */
    div[data-testid="stDownloadButton"] button {
        border-radius: 12px !important;
        border: 1px solid rgba(52, 211, 153, 0.35) !important;
        background: rgba(5, 150, 105, 0.15) !important;
        color: #34d399 !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background: rgba(5, 150, 105, 0.3) !important;
        border-color: #10b981 !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }

    /* Tables */
    table {
        width: 100% !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        margin: 18px 0 !important;
        font-size: 0.92rem !important;
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
        background: rgba(15, 23, 42, 0.5) !important;
    }
    th {
        background: linear-gradient(135deg, rgba(46, 16, 101, 0.7) 0%, rgba(30, 27, 75, 0.8) 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        padding: 13px 18px !important;
        border-bottom: 2px solid rgba(168, 85, 247, 0.4) !important;
        text-align: left !important;
    }
    td {
        padding: 12px 18px !important;
        border-bottom: 1px solid rgba(168, 85, 247, 0.14) !important;
        color: #e2e8f0 !important;
    }
    tr:last-child td { border-bottom: none !important; }
    tr:nth-child(even) { background-color: rgba(255, 255, 255, 0.02) !important; }

    /* Floating Chat Input */
    div[data-testid="stChatInput"] {
        border-radius: 22px !important;
        border: 1.5px solid rgba(168, 85, 247, 0.4) !important;
        background: rgba(18, 19, 31, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #a855f7 !important;
        box-shadow: 0 8px 36px rgba(168, 85, 247, 0.4) !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        font-size: 0.96rem !important;
    }

    /* Citation Badges */
    .source-box {
        margin-top: 16px;
        padding: 12px 16px;
        background: rgba(168, 85, 247, 0.08);
        border: 1px dashed rgba(168, 85, 247, 0.35);
        border-radius: 12px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
    }
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(168, 85, 247, 0.18);
        border: 1px solid rgba(168, 85, 247, 0.4);
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.77rem;
        color: #e9d5ff;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Multi-Chat Session State Initialization (ChatGPT Style)
# ============================================================
if "conversations" not in st.session_state:
    st.session_state.conversations = load_all_conversations()

if not st.session_state.conversations:
    init_id = f"chat_{int(time.time())}"
    st.session_state.conversations[init_id] = {
        "id": init_id,
        "title": "New Conversation",
        "messages": [],
        "created_at": time.time(),
        "branch": "CSE",
        "year": "1st Year",
    }
    st.session_state.current_chat_id = init_id
    save_all_conversations(st.session_state.conversations)

if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.conversations:
    st.session_state.current_chat_id = list(st.session_state.conversations.keys())[0]

# Active conversation reference
current_convo = st.session_state.conversations[st.session_state.current_chat_id]

if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = {
        "active": False,
        "subject": "",
        "questions": [],
        "current_idx": 0,
        "score": 0,
    }

if "search_query_filter" not in st.session_state:
    st.session_state.search_query_filter = ""

# ============================================================
# Left Panel: ChatGPT-Style Chat History & More Options
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <span style="font-size: 1.4rem;">🎓</span>
            <div>
                <h3 class="sidebar-brand-title">ABES Campus AI</h3>
                <span style="font-size: 0.72rem; color: #94a3b8;">Conversational Intelligence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. New Chat Button
    if st.button("➕ New Chat", use_container_width=True, key="new_chat_main"):
        new_id = f"chat_{int(time.time())}"
        st.session_state.conversations[new_id] = {
            "id": new_id,
            "title": "New Conversation",
            "messages": [],
            "created_at": time.time(),
            "branch": current_convo.get("branch", "CSE"),
            "year": current_convo.get("year", "1st Year"),
        }
        st.session_state.current_chat_id = new_id
        st.session_state.quiz_state = {"active": False, "subject": "", "questions": [], "current_idx": 0, "score": 0}
        st.session_state.pending_clarification = None
        save_all_conversations(st.session_state.conversations)
        st.rerun()

    # 2. Search History Bar
    st.session_state.search_query_filter = st.text_input(
        "Search History",
        placeholder="🔍 Search past chats...",
        label_visibility="collapsed",
    )

    # 3. Chat History List
    st.markdown('<div class="sidebar-section-title">Recent Conversations</div>', unsafe_allow_html=True)

    sorted_chat_ids = sorted(
        st.session_state.conversations.keys(),
        key=lambda k: st.session_state.conversations[k].get("created_at", 0),
        reverse=True,
    )

    q_filter = st.session_state.search_query_filter.strip().lower()
    visible_chats = []
    for cid in sorted_chat_ids:
        title = st.session_state.conversations[cid].get("title", "New Conversation")
        if not q_filter or q_filter in title.lower():
            visible_chats.append((cid, title))

    if visible_chats:
        for cid, title in visible_chats[:12]:
            is_active = (cid == st.session_state.current_chat_id)
            btn_class = "convo-btn-active" if is_active else "convo-btn"
            icon = "💬" if not is_active else "✨"

            col_chat, col_del = st.columns([0.84, 0.16])
            with col_chat:
                st.markdown(f'<div class="{btn_class}">', unsafe_allow_html=True)
                label = f"{icon} {title[:22]}" + ("..." if len(title) > 22 else "")
                if st.button(label, key=f"chat_select_{cid}", use_container_width=True):
                    st.session_state.current_chat_id = cid
                    st.session_state.quiz_state = {"active": False, "subject": "", "questions": [], "current_idx": 0, "score": 0}
                    st.session_state.pending_clarification = None
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            with col_del:
                st.markdown('<div class="del-btn">', unsafe_allow_html=True)
                if st.button("✕", key=f"del_{cid}", help="Delete this chat"):
                    del st.session_state.conversations[cid]
                    if not st.session_state.conversations:
                        fallback_id = f"chat_{int(time.time())}"
                        st.session_state.conversations[fallback_id] = {
                            "id": fallback_id,
                            "title": "New Conversation",
                            "messages": [],
                            "created_at": time.time(),
                            "branch": "CSE",
                            "year": "1st Year",
                        }
                        st.session_state.current_chat_id = fallback_id
                    elif st.session_state.current_chat_id == cid:
                        st.session_state.current_chat_id = list(st.session_state.conversations.keys())[0]
                    save_all_conversations(st.session_state.conversations)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.caption("No conversations found matching search.")

    st.markdown("---")

    # 4. Student Profile Settings
    st.markdown('<div class="sidebar-section-title">Academic Profile</div>', unsafe_allow_html=True)

    branches = ["General / All", "CSE", "IT", "ECE", "ME", "EE", "Civil", "CSE-AIML", "CSE-DS"]
    cur_branch = current_convo.get("branch", "CSE")
    b_idx = branches.index(cur_branch) if cur_branch in branches else 1

    selected_branch = st.selectbox(
        "Branch / Department",
        branches,
        index=b_idx,
        help="Contextualizes answers for your specific curriculum.",
    )
    current_convo["branch"] = selected_branch

    years = ["All Years", "1st Year", "2nd Year", "3rd Year", "4th Year"]
    cur_year = current_convo.get("year", "1st Year")
    y_idx = years.index(cur_year) if cur_year in years else 1

    selected_year = st.selectbox(
        "Academic Year",
        years,
        index=y_idx,
        help="Filters fee schedules and subject units.",
    )
    current_convo["year"] = selected_year

    st.markdown('<div class="sidebar-section-title">Knowledge Sources</div>', unsafe_allow_html=True)
    live_web_enabled = st.toggle(
        "🌐 Live abes.ac.in Search",
        value=True,
        help="Searches official abes.ac.in website when information is not found in local documents.",
        key="live_web_toggle",
    )

    # 5. Export and Reset Controls
    st.markdown('<div class="sidebar-section-title">Session Management</div>', unsafe_allow_html=True)
    col_exp, col_clear = st.columns(2)
    with col_exp:
        if current_convo["messages"]:
            transcript = f"# ABES Campus AI - {current_convo['title']}\n\n"
            for m in current_convo["messages"]:
                speaker = "👤 Student" if m["role"] == "user" else "🎓 ABES AI"
                transcript += f"**{speaker}:**\n{m['content']}\n\n---\n\n"
            st.download_button(
                "📥 Export",
                data=transcript,
                file_name=f"{current_convo['title'][:20].replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.button("📥 Export", disabled=True, use_container_width=True)

    with col_clear:
        if st.button("🗑️ Reset", use_container_width=True):
            st.session_state.conversations = {}
            fallback_id = f"chat_{int(time.time())}"
            st.session_state.conversations[fallback_id] = {
                "id": fallback_id,
                "title": "New Conversation",
                "messages": [],
                "created_at": time.time(),
                "branch": "CSE",
                "year": "1st Year",
            }
            st.session_state.current_chat_id = fallback_id
            save_all_conversations(st.session_state.conversations)
            st.rerun()

    with st.expander("Official Campus Links"):
        st.markdown(
            """
        - [ABES Official Website](https://abes.ac.in)
        - [AKTU Student ERP](https://erp.aktu.ac.in)
        - [AKTU OneView Results](https://erp.aktu.ac.in/WebPages/OneView/OneView.aspx)
        """
        )

# ============================================================
# Brand Hero Header (With exact tagline and pill tags)
# ============================================================
st.markdown(
    """
<div class="brand-hero">
    <div class="brand-badge">
        <span class="live-dot"></span>
        Official Campus Intelligence Portal
    </div>
    <h1 class="brand-title">🎓 ABES Campus AI</h1>
    <p class="brand-sub">
        Instant, verified academic intelligence for <b>Admissions</b>, <b>Hostel & Mess</b>, 
        <b>AKTU Syllabus</b>, and <b>Examination Papers</b>.
    </p>
    <div class="pill-tags">
        <span class="pill-tag">🏛️ AKTU Code: 032</span>
        <span class="pill-tag">📚 Syllabus & PYQ Vault</span>
        <span class="pill-tag">🎯 Interactive Viva Prep</span>
        <span class="pill-tag">⚡ 24/7 Verified Helpdesk</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# System Loader + Automatic ChromaDB Ingestion
# ============================================================
@st.cache_resource
def load_system():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="db")
    collection = client.get_or_create_collection(name="university")

    # Detect which files have already been indexed in ChromaDB
    existing_sources = set()
    try:
        existing_data = collection.get(include=["metadatas"])
        for m in existing_data.get("metadatas", []):
            if m and "source" in m:
                existing_sources.add(m["source"])
    except Exception:
        pass

    new_docs, new_metas, new_ids = [], [], []

    # 1. Ingest any new/unindexed PDFs in data/
    if os.path.exists(DATA_FOLDER):
        pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".pdf")]
        for filename in pdf_files:
            if filename in existing_sources:
                continue

            filepath = os.path.join(DATA_FOLDER, filename)
            try:
                reader = PdfReader(filepath)
                full_text = ""
                for page in reader.pages:
                    txt = page.extract_text() or ""
                    if txt.strip():
                        full_text += txt + "\n"

                chunk_size = 1200
                overlap = 200
                start, c_num = 0, 0
                while start < len(full_text):
                    end = start + chunk_size
                    chunk = full_text[start:end].strip()
                    if chunk:
                        enriched = f"Institution: ABES Engineering College\nDocument: {filename}\nContent:\n{chunk}"
                        new_docs.append(enriched)
                        new_metas.append({"source": filename, "type": "pdf"})
                        new_ids.append(f"pdf_{filename}_{c_num}_{int(time.time())}")
                        c_num += 1
                    start = end - overlap
            except Exception:
                pass

    # 2. Ingest any unindexed JSON dataset files in dataset/
    SKIP_INTENTS = {"greeting", "goodbye", "creator", "name", "random", "swear", "salutaion", "task"}
    if os.path.exists(DATASET_FOLDER):
        json_files = [f for f in os.listdir(DATASET_FOLDER) if f.lower().endswith(".json")]
        for filename in json_files:
            if filename in existing_sources:
                continue

            filepath = os.path.join(DATASET_FOLDER, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    data = json.load(file)
                intents = data.get("intents", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])

                for index, item in enumerate(intents):
                    intent = item.get("intent", "")
                    if intent.lower() in SKIP_INTENTS:
                        continue
                    questions = item.get("text", [])
                    responses = item.get("responses", [])

                    content = f"Intent: {intent}\n"
                    if questions:
                        content += "Example questions:\n" + "\n".join(f"- {q}" for q in questions) + "\n"
                    if responses:
                        content += "Dataset responses:\n" + "\n".join(f"- {r}" for r in responses)

                    new_docs.append(content)
                    new_metas.append({"source": filename, "type": "json", "intent": intent})
                    new_ids.append(f"json_{filename}_{index}")
            except Exception:
                pass

    if new_docs:
        embeddings = model.encode(new_docs, show_progress_bar=False).tolist()
        collection.add(documents=new_docs, embeddings=embeddings, metadatas=new_metas, ids=new_ids)

    return model, collection


model, collection = load_system()

# ------------------------------------------------------------
# Safe API Key Loading & Dynamic Model Detection
# ------------------------------------------------------------
try:
    secret_key = st.secrets.get("GROQ_API_KEY", None)
except Exception:
    secret_key = None

api_key = os.environ.get("GROQ_API_KEY") or secret_key
if not api_key:
    st.error("🔑 GROQ_API_KEY is not set. Please add it to your Streamlit Cloud Secrets or environment.")
    st.stop()

ai = Groq(api_key=api_key)


def get_working_models(client):
    candidates = [
        "llama-3.3-70b-versatile",
        "llama3-8b-8192",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    try:
        available_ids = [m.id for m in client.models.list().data]
        valid = [c for c in candidates if c in available_ids]
        for m in available_ids:
            if m not in valid and not any(x in m for x in ["whisper", "guard", "embed"]):
                valid.append(m)
        if valid:
            return valid
    except Exception:
        pass
    return ["openai/gpt-oss-20b", "llama3-8b-8192", "llama-3.3-70b-versatile"]


WORKING_MODELS = get_working_models(ai)
ACTIVE_MODEL = WORKING_MODELS[0]


def safe_chat_completion(client, messages, temperature=0.2, max_tokens=2200, stream=False):
    """Crash-proof chat completion that falls back automatically if a model is deprecated."""
    last_err = None
    for model_name in WORKING_MODELS:
        try:
            return client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
        except Exception as e:
            err_str = str(e).lower()
            if "model_not_found" in err_str or "does not exist" in err_str or "404" in err_str:
                last_err = e
                continue
            raise e
    if last_err:
        raise last_err


SEMESTER_PATTERN = re.compile(
    r"\b(1st|2nd|3rd|4th|5th|6th|7th|8th|first|second|third|fourth|fifth|sixth|seventh|eighth)\s*sem(ester)?\b"
    r"|\bsem(ester)?\s*[1-8]\b",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(
    r"\b(1st|2nd|3rd|4th|first|second|third|fourth)\s*year\b",
    re.IGNORECASE,
)


def has_semester(text):
    return bool(SEMESTER_PATTERN.search(text))


def has_year(text):
    return bool(YEAR_PATTERN.search(text))


# ============================================================
# Live Website Search & Scraping Helpers (abes.ac.in)
# ============================================================
def search_abes_website(query, max_results=3):
    if not WEB_SEARCH_AVAILABLE:
        return []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"site:{ABES_DOMAIN} {query}", max_results=max_results))
            if not results:
                results = list(ddgs.text(f"ABES Engineering College Ghaziabad {query}", max_results=max_results))
            return results
    except Exception:
        return []


def fetch_page_text(url, max_chars=2500):
    if not WEB_SEARCH_AVAILABLE or not url:
        return ""
    try:
        resp = requests.get(
            url,
            timeout=4,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "svg"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ").split())
            return text[:max_chars]
    except Exception:
        pass
    return ""


# ============================================================
# Active Prompt Holder (Captures either button click or chat input directly)
# ============================================================
active_prompt = None

# ============================================================
# Empty State: Starter Prompt Cards (Only visible when no messages)
# ============================================================
if not current_convo["messages"]:
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <p style="color: #c084fc; font-size: 0.95rem; font-weight: 600; margin-bottom: 12px;">
                💡 Select a prompt to start exploring verified information or type your question below:
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    starter_cards = [
        ("Hostel & Mess Fees", "What is the complete hostel fee structure for AC and Non-AC rooms?"),
        ("B.Tech Admission Fees", "What is the detailed B.Tech 1st year admission fee schedule?"),
        ("DBMS Course Syllabus", "What is the complete syllabus and units for DBMS?"),
        ("Previous Year Papers", "Download previous year question papers for exams"),
        ("Viva Knowledge Check", "Quiz me with 3 MCQ viva questions on Operating Systems"),
        ("Placements & Recruiters", "What are the top placement recruiters and average packages?"),
    ]

    c1, c2 = st.columns(2)
    for i, (title, prompt_val) in enumerate(starter_cards):
        col = c1 if i % 2 == 0 else c2
        if col.button(f"📌 {title}", key=f"starter_card_{i}", use_container_width=True):
            active_prompt = prompt_val


# ============================================================
# Render Active Chat Conversation History
# ============================================================
for idx, message in enumerate(current_convo["messages"]):
    is_user = message["role"] == "user"
    avatar_icon = "👤" if is_user else "🎓"
    hook_class = "user-hook" if is_user else "bot-hook"

    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(f'<div class="{hook_class}"></div>', unsafe_allow_html=True)
        st.markdown(message["content"])

        if message.get("pyq_files"):
            st.markdown("---")
            st.caption("Available Examination Papers:")
            folder = message.get("pyq_folder", QUESTION_PAPERS_FOLDER)
            for f_idx, pdf_file in enumerate(message["pyq_files"]):
                filepath = os.path.join(folder, pdf_file)
                if os.path.exists(filepath):
                    with open(filepath, "rb") as f:
                        file_bytes = f.read()
                    subject_name = pdf_file.rsplit(".", 1)[0].replace("_", " ").title()
                    st.download_button(
                        label=f"📄 Download {subject_name} Paper (PDF)",
                        data=file_bytes,
                        file_name=pdf_file,
                        mime="application/pdf",
                        key=f"dl_{st.session_state.current_chat_id}_{idx}_{f_idx}_{pdf_file}",
                        use_container_width=True,
                    )


# ============================================================
# In-Chat Interactive Quiz UI
# ============================================================
quiz = st.session_state.quiz_state
if quiz["active"] and quiz["current_idx"] < len(quiz["questions"]):
    curr_q = quiz["questions"][quiz["current_idx"]]
    q_num = quiz["current_idx"] + 1
    total_q = len(quiz["questions"])

    with st.chat_message("assistant", avatar="🎓"):
        st.markdown('<div class="bot-hook"></div>', unsafe_allow_html=True)
        st.markdown(f"**Viva Question {q_num} of {total_q}:** {curr_q['question']}")

        c1, c2 = st.columns(2)
        cols = [c1, c2, c1, c2]

        user_choice = None
        for i, opt in enumerate(curr_q["options"]):
            letter = opt.split(")")[0].strip()
            if cols[i].button(opt, key=f"quiz_opt_{q_num}_{i}", use_container_width=True):
                user_choice = letter

        if st.button("Quit Quiz", key="quit_quiz_btn"):
            st.session_state.quiz_state["active"] = False
            st.rerun()

        if user_choice:
            is_correct = user_choice.upper() == curr_q["correct_option"].upper()
            if is_correct:
                feedback = f"✅ **Correct!** ({curr_q['correct_option']})\n\n{curr_q['explanation']}"
                st.session_state.quiz_state["score"] += 1
            else:
                feedback = f"❌ **Incorrect.** You selected ({user_choice}), but the correct answer is **({curr_q['correct_option']})**.\n\n{curr_q['explanation']}"

            current_convo["messages"].append({"role": "user", "content": f"Selected ({user_choice})"})
            current_convo["messages"].append({"role": "assistant", "content": feedback})

            st.session_state.quiz_state["current_idx"] += 1
            if st.session_state.quiz_state["current_idx"] >= total_q:
                final_score = st.session_state.quiz_state["score"]
                summary = f"🎉 **Quiz Complete.** Final Score: **{final_score} / {total_q}**"
                current_convo["messages"].append({"role": "assistant", "content": summary})
                st.session_state.quiz_state["active"] = False

            save_all_conversations(st.session_state.conversations)
            st.rerun()


# ============================================================
# Chat Input Bar (Checks input directly)
# ============================================================
user_typed = st.chat_input("Ask a question about admissions, hostels, exams, or syllabus...")
if user_typed:
    active_prompt = user_typed

# ============================================================
# Unified Pipeline (Processes BOTH Button Clicks and Chat Input)
# ============================================================
if active_prompt:
    user_input = active_prompt

    if current_convo["title"] == "New Conversation":
        current_convo["title"] = user_input[:28] + ("..." if len(user_input) > 28 else "")

    with st.chat_message("user", avatar="👤"):
        st.markdown('<div class="user-hook"></div>', unsafe_allow_html=True)
        st.markdown(user_input)
    current_convo["messages"].append({"role": "user", "content": user_input})

    q_lower = user_input.lower().strip()

    # Dynamic student profile extraction
    for branch_name in ["cse", "it", "ece", "me", "ee", "civil"]:
        if f" {branch_name} " in f" {q_lower} ":
            current_convo["branch"] = branch_name.upper()
    for year_name in ["1st year", "2nd year", "3rd year", "4th year"]:
        if year_name in q_lower:
            current_convo["year"] = year_name

    # 1. Previous Year Papers (Checked first before datesheet logic)
    pyq_triggers = [
        "previous year", "pyq", "pyqs", "question paper", "ques paper", "old question",
        "old paper", "past paper", "exam paper", "purane paper", "sample paper", "papers"
    ]
    if any(t in q_lower for t in pyq_triggers):
        folder = get_question_papers_folder()
        all_pdfs = sorted([f for f in os.listdir(folder) if f.lower().endswith(".pdf")]) if os.path.exists(folder) else []

        if all_pdfs:
            stop_words = {"the", "for", "and", "pyq", "pyqs", "ques", "question", "questions", "paper", "papers", "previous", "year", "old", "past", "exam", "give", "me", "show", "i", "want", "of", "in"}
            user_tokens = [w for w in re.findall(r'\b\w+\b', q_lower) if w not in stop_words and len(w) > 2]
            matched = [p for p in all_pdfs if any(t in p.lower() for t in user_tokens)] if user_tokens else []
            selected_pdfs = matched if matched else all_pdfs

            content = f"Here are the previous year examination papers available for download (**{len(selected_pdfs)} paper(s)**):"
            current_convo["messages"].append({
                "role": "assistant",
                "content": content,
                "pyq_files": selected_pdfs,
                "pyq_folder": folder
            })
        else:
            current_convo["messages"].append({
                "role": "assistant",
                "content": f"No question papers found in `{folder}/`."
            })
        save_all_conversations(st.session_state.conversations)
        st.rerun()

    # 2. Practice Quiz
    quiz_triggers = ["quiz", "test me", "viva question", "ask me question", "mcq"]
    if any(t in q_lower for t in quiz_triggers):
        query_emb = model.encode(user_input).tolist()
        results = collection.query(query_embeddings=[query_emb], n_results=4, include=["documents"])
        docs = results.get("documents", [[]])[0]
        context = "\n".join(docs)

        quiz_prompt = f"""
Create 3 high-quality multiple choice questions based on: {user_input}.
Context:
{context[:2000]}

Respond ONLY with valid JSON array:
[
  {{
    "question": "Question text here?",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct_option": "A",
    "explanation": "Brief explanation."
  }}
]
"""
        try:
            with st.spinner("Preparing viva questions..."):
                res = safe_chat_completion(
                    ai,
                    messages=[{"role": "user", "content": quiz_prompt}],
                    temperature=0.3,
                    max_tokens=900,
                )
            raw_text = res.choices[0].message.content.strip()
            match = re.search(r"\[.*\]", raw_text, re.DOTALL)
            if match:
                questions = json.loads(match.group())
                st.session_state.quiz_state = {
                    "active": True,
                    "subject": user_input.title(),
                    "questions": questions,
                    "current_idx": 0,
                    "score": 0,
                }
                current_convo["messages"].append({"role": "assistant", "content": f"Starting practice quiz on **{user_input.title()}**."})
                save_all_conversations(st.session_state.conversations)
                st.rerun()
        except Exception:
            pass

    # 3. Clarification handling
    if st.session_state.pending_clarification:
        clar = st.session_state.pending_clarification
        user_input = f"{clar['original']} for {user_input}"
        q_lower = user_input.lower()
        st.session_state.pending_clarification = None

    else:
        # Only clarify for general datesheet questions if semester is completely missing
        datesheet_words = ["datesheet", "date sheet", "when are exams", "schedule of exams"]
        if any(w in q_lower for w in datesheet_words) and not has_semester(q_lower):
            answer = "Which semester's examination schedule are you asking about? (e.g., 1st, 3rd, 5th, or 7th semester)"
            current_convo["messages"].append({"role": "assistant", "content": answer})
            st.session_state.pending_clarification = {"original": user_input}
            save_all_conversations(st.session_state.conversations)
            st.rerun()

    # 4. Semantic Search (Dual-Source: Local PDFs + Official Website)
    acronym_map = {
        r'\bdbms\b': 'DBMS Database Management System',
        r'\bos\b': 'OS Operating Systems',
        r'\bcn\b': 'CN Computer Networks',
        r'\bdsa?\b': 'DSA Data Structures and Algorithms',
        r'\bdaa\b': 'DAA Design and Analysis of Algorithms',
        r'\bcoa\b': 'COA Computer Organization and Architecture',
        r'\bse\b': 'SE Software Engineering',
        r'\bwt\b': 'WT Web Technology',
        r'\btafl\b': 'TAFL Theory of Automata and Formal Languages',
        r'\bcompiler\b': 'Compiler Design CD',
        r'\bai\b': 'AI Artificial Intelligence',
        r'\bml\b': 'ML Machine Learning',
        r'\boops?\b': 'OOPs Object Oriented Programming',
    }

    # Normalize syllabus typos (e.g. sylabus, syllabus, syllubus, silabus)
    is_syllabus_query = bool(re.search(r'\b(syll?ab[uo]s|curriculum|course\s*outline|units?|topics?)\b', q_lower))

    search_query = user_input
    for pat, expansion in acronym_map.items():
        if re.search(pat, q_lower):
            search_query += f" {expansion}"

    if is_syllabus_query:
        search_query += " syllabus units course code topics curriculum detailed"
    elif "hostel" in q_lower or "mess" in q_lower:
        search_query += " hostel fee structure lodging boarding room rent mess charges security deposit AC Non-AC"
    elif "admission" in q_lower or "tuition" in q_lower or "college fee" in q_lower or "btech fee" in q_lower or "fee" in q_lower:
        search_query += " academic fee tuition fee admission schedule of charges b.tech"
    elif "exam" in q_lower or "schedule" in q_lower:
        search_query += " academic calendar examination schedule sessional test end sem"
    elif "placement" in q_lower or "package" in q_lower:
        search_query += " placement statistics packages highest average companies recruitment"

    # Step A: Retrieve from Local PDF Documents (ChromaDB)
    documents = []
    sources = set()
    try:
        emb = model.encode(search_query).tolist()
        results = collection.query(
            query_embeddings=[emb],
            n_results=6,
            include=["documents", "metadatas"],
        )
        raw_docs = results.get("documents", [[]])[0]
        raw_meta = results.get("metadatas", [[]])[0]

        for doc, meta in zip(raw_docs, raw_meta):
            if doc and doc.strip():
                documents.append(doc)
                if meta and "source" in meta:
                    sources.add(meta["source"])
    except Exception:
        pass

    # Step B: Retrieve from Official College Website (abes.ac.in)
    web_context_parts = []
    web_sources = []
    is_live_enabled = st.session_state.get("live_web_toggle", True)

    with st.chat_message("assistant", avatar="🎓"):
        st.markdown('<div class="bot-hook"></div>', unsafe_allow_html=True)

        if WEB_SEARCH_AVAILABLE and is_live_enabled:
            with st.spinner("🔍 Consulting college documents & official abes.ac.in website..."):
                web_results = search_abes_website(user_input, max_results=2)
                for r in web_results:
                    page_url = r.get("href", "")
                    page_title = r.get("title", "Official Page")
                    snippet = r.get("body", "")
                    page_text = fetch_page_text(page_url) if page_url else ""
                    chosen_text = page_text if len(page_text) > 100 else snippet

                    if chosen_text and chosen_text.strip():
                        web_context_parts.append(f"Source URL: {page_url}\nTitle: {page_title}\nContent:\n{chosen_text}")
                        if page_url and page_url not in web_sources:
                            web_sources.append(page_url)

        # Step C: Combine Context from BOTH PDF Documents and Official Website
        context_blocks = []
        if documents:
            context_blocks.append("### FROM COLLEGE DOCUMENTS (PDFs):\n" + "\n\n---\n\n".join(documents))
        if web_context_parts:
            context_blocks.append("### FROM OFFICIAL COLLEGE WEBSITE (abes.ac.in):\n" + "\n\n---\n\n".join(web_context_parts))

        exam_triggers = ["exam", "exams", "schedule", "date sheet", "datesheet", "when are exams", "exam date"]
        is_exam_query = any(trigger in q_lower for trigger in exam_triggers)
        is_placement_query = any(trigger in q_lower for trigger in ["placement", "package", "packages", "salary", "companies", "recruiters"])

        if context_blocks or is_exam_query or is_placement_query:
            combined_context = "\n\n====================\n\n".join(context_blocks)
            student_ctx = f"Branch: {current_convo.get('branch', 'General')}, Year: {current_convo.get('year', 'All')}"

            system_prompt = f"""
You are the official AI Assistant for ABES Engineering College (AKTU).
Tone: Comprehensive, clear, structured, and student-friendly.
Student Profile: {student_ctx}

INSTRUCTIONS:
1. Synthesize accurate facts from BOTH the college PDF documents and the official website records provided below.
2. SYLLABUS: When asked for a subject syllabus (e.g. DBMS, OS, Computer Networks, Data Structures, etc.):
   - Provide the complete syllabus units (Unit 1, Unit 2, Unit 3, Unit 4, Unit 5).
   - List key topics, course code, and practical/lab modules if present.
3. FEES: Break down fees (tuition, hostel, mess, AC vs Non-AC, seaters, deposits) in clean Markdown tables.
4. EXAMS: Present the AKTU semester cycles (Odd sem: Dec–Jan, Even sem: May–June) or specific dates if in documents.
5. If details are present in the documents, state them completely without withholding information.
"""

            messages = [{"role": "system", "content": system_prompt}]
            for msg in current_convo["messages"][-4:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({
                "role": "user",
                "content": f"University Information (PDFs & Website):\n{combined_context}\n\nStudent Question: {user_input}",
            })

            try:
                stream = safe_chat_completion(
                    ai,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2200,
                    stream=True,
                )

                def chunk_generator():
                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content

                full_answer = st.write_stream(chunk_generator())

                # Show verified citations for BOTH PDF sources and Website sources
                citation_chips = []
                if sources:
                    for s in sorted(sources):
                        citation_chips.append(f"<span class='source-chip'>📄 PDF: {s}</span>")
                if web_sources:
                    for ws in web_sources[:3]:
                        citation_chips.append(f"<a href='{ws}' target='_blank' style='display: inline-block; margin-right: 8px;'><span class='source-chip' style='color: #60a5fa;'>🔗 Web: {ws}</span></a>")

                if citation_chips:
                    citation_html = f"""
                    <div class="source-box">
                        <span style="font-size: 0.76rem; font-weight: 700; color: #d8b4fe;">VERIFIED SOURCES (PDF & WEB):</span><br>
                        {" ".join(citation_chips)}
                    </div>
                    """
                    st.markdown(citation_html, unsafe_allow_html=True)
                    source_text_list = [f"`📄 {s}`" for s in sorted(sources)] + [f"[{ws}]({ws})" for ws in web_sources]
                    full_answer += "\n\n**Sources:** " + ", ".join(source_text_list)

            except Exception:
                res = safe_chat_completion(
                    ai,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2200,
                )
                full_answer = res.choices[0].message.content
                st.markdown(full_answer)

        else:
            full_answer = "This information does not appear to be present in the available university documents or the official website. Please check with the college registrar office or visit abes.ac.in directly."
            st.markdown(full_answer)

    current_convo["messages"].append({"role": "assistant", "content": full_answer})
    save_all_conversations(st.session_state.conversations)
    st.rerun()

# ============================================================
# Footer
# ============================================================
st.markdown(
    """
<div style="text-align: center; margin-top: 36px; padding: 10px;">
    <p style="color: #64748b; font-size: 0.76rem; margin: 0;">
        🛡️ ABES Campus AI is an intelligent assistant. Please verify official deadlines, fee receipts, and dates with the college registrar.
    </p>
</div>
""",
    unsafe_allow_html=True,
)
