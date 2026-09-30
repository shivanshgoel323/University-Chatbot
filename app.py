import json
import os
import re
import time

import chromadb
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit as st

# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="ABES University Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

DATA_FOLDER = "data"
ACTIVE_MODEL = "openai/gpt-oss-20b"

def get_question_papers_folder():
    candidates = ["question_papers", "question_paper", "questionpapers", "pyq", "pyqs", "data/question_papers"]
    for folder in candidates:
        if os.path.exists(folder) and os.path.isdir(folder):
            return folder
    return "question_papers"

QUESTION_PAPERS_FOLDER = get_question_papers_folder()

# ------------------------------------------------------------
# Fascinating Dual-Color Theme CSS (Emerald Jade & Royal Amethyst)
# ------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Clearance at the top to prevent cropped header */
    .block-container {
        padding-top: 4.2rem !important;
        padding-bottom: 4.5rem !important;
        max-width: 820px !important;
    }

    /* Keyframe Animations */
    @keyframes bubbleIn {
        from { opacity: 0; transform: translateY(10px) scale(0.99); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* Brand Header */
    .brand-container {
        text-align: center;
        margin-top: 0.5rem !important;
        margin-bottom: 2.2rem !important;
        animation: bubbleIn 0.35s ease-out forwards;
    }
    .brand-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(168, 85, 247, 0.18) 100%);
        color: #10b981;
        border: 1px solid rgba(52, 211, 153, 0.35);
        border-radius: 50px;
        padding: 5px 16px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 12px;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.12);
    }
    .brand-title {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        letter-spacing: -0.6px;
    }
    .brand-sub {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 6px;
    }

    /* ============================================================ */
    /* 1. USER INPUT BUBBLE: Rich Emerald Jade Jewel Gradient       */
    /* ============================================================ */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]),
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #065f46 0%, #047857 50%, #0d9488 100%) !important;
        border: 1px solid rgba(52, 211, 153, 0.45) !important;
        border-radius: 20px 20px 4px 20px !important;
        box-shadow: 0 4px 18px rgba(4, 120, 87, 0.28), 0 1px 3px rgba(0, 0, 0, 0.15) !important;
        padding: 16px 20px !important;
        margin-bottom: 1.1rem !important;
        animation: bubbleIn 0.25s ease-out forwards;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) * {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* ============================================================ */
    /* 2. ASSISTANT OUTPUT BUBBLE: Royal Amethyst / Velvet Iris     */
    /* ============================================================ */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]),
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.92) 0%, rgba(46, 16, 101, 0.85) 60%, rgba(20, 20, 42, 0.95) 100%) !important;
        border: 1px solid rgba(168, 85, 247, 0.35) !important;
        border-left: 4.5px solid #a855f7 !important;
        border-radius: 4px 20px 20px 20px !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25), 0 0 16px rgba(168, 85, 247, 0.12) !important;
        padding: 18px 22px !important;
        margin-bottom: 1.3rem !important;
        animation: bubbleIn 0.25s ease-out forwards;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) * {
        color: #f8fafc !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) p {
        line-height: 1.65 !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) strong {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Light Mode Fallback for Assistant Bubble */
    @media (prefers-color-scheme: light) {
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]),
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 60%, #ede9fe 100%) !important;
            border: 1px solid rgba(147, 51, 234, 0.25) !important;
            border-left: 4.5px solid #9333ea !important;
            box-shadow: 0 4px 20px rgba(147, 51, 234, 0.08) !important;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) * {
            color: #1e1b4b !important;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) strong {
            color: #0f0a2a !important;
        }
    }

    /* ============================================================ */
    /* 3. Fascinating Starter Topic Cards                          */
    /* ============================================================ */
    div[data-testid="stButton"] button {
        border-radius: 14px !important;
        border: 1px solid rgba(168, 85, 247, 0.25) !important;
        background: rgba(168, 85, 247, 0.05) !important;
        padding: 14px 18px !important;
        text-align: left !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        color: inherit !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-testid="stButton"] button:hover {
        border-color: #a855f7 !important;
        background: rgba(168, 85, 247, 0.14) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.18) !important;
        color: #d8b4fe !important;
    }

    /* Download Buttons */
    div[data-testid="stDownloadButton"] button {
        border-radius: 12px !important;
        border: 1px solid rgba(52, 211, 153, 0.35) !important;
        background: rgba(5, 150, 105, 0.1) !important;
        color: inherit !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background: rgba(5, 150, 105, 0.22) !important;
        border-color: #10b981 !important;
        transform: translateY(-1px) !important;
    }

    /* ============================================================ */
    /* 4. Glowing Chat Input Bar                                    */
    /* ============================================================ */
    div[data-testid="stChatInput"] {
        border-radius: 18px !important;
        border: 1.5px solid rgba(168, 85, 247, 0.3) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #a855f7 !important;
        box-shadow: 0 4px 25px rgba(168, 85, 247, 0.28) !important;
    }

    /* ============================================================ */
    /* 5. Crystal-Clear Tables (Dark & Light Mode Contrast)        */
    /* ============================================================ */
    table {
        width: 100% !important;
        border-collapse: collapse !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        margin: 14px 0 !important;
        font-size: 0.9rem !important;
        border: 1px solid rgba(168, 85, 247, 0.25) !important;
        background: transparent !important;
    }
    th {
        background-color: rgba(168, 85, 247, 0.18) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        padding: 11px 14px !important;
        border-bottom: 2px solid rgba(168, 85, 247, 0.35) !important;
        text-align: left !important;
    }
    td {
        padding: 10px 14px !important;
        border-bottom: 1px solid rgba(168, 85, 247, 0.12) !important;
        color: #f1f5f9 !important;
    }
    tr:nth-child(even) {
        background-color: rgba(255, 255, 255, 0.03) !important;
    }

    @media (prefers-color-scheme: light) {
        th {
            background-color: #ede9fe !important;
            color: #1e1b4b !important;
            border-bottom: 2px solid #c4b5fd !important;
        }
        td {
            border-bottom: 1px solid #e9d5ff !important;
            color: #2e1065 !important;
        }
        tr:nth-child(even) {
            background-color: rgba(237, 233, 254, 0.4) !important;
        }
    }

    /* Source Tags */
    .source-tag {
        display: inline-block;
        background: rgba(168, 85, 247, 0.15);
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 6px;
        padding: 3px 9px;
        font-size: 0.76rem;
        color: #c084fc;
        font-family: monospace;
        margin-right: 6px;
        margin-top: 5px;
        font-weight: 600;
    }
    @media (prefers-color-scheme: light) {
        .source-tag {
            background: #ede9fe !important;
            border: 1px solid #d8b4fe !important;
            color: #6b21a8 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Header (Centered & Fascinating)
# ------------------------------------------------------------
st.markdown("""
<div class="brand-container">
    <div class="brand-badge">⚡ Official AI Campus Portal • 2026-27</div>
    <h1 class="brand-title">🎓 ABES University Assistant</h1>
    <p class="brand-sub">Instant verified answers for admissions, hostels, exams & syllabus</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# System Loader + Automatic Cloud Ingest
# ------------------------------------------------------------
@st.cache_resource
def load_system():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="db")
    collection = client.get_or_create_collection(name="university")

    if collection.count() == 0 and os.path.exists(DATA_FOLDER):
        pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".pdf")]
        docs, metas, ids = [], [], []

        for filename in pdf_files:
            filepath = os.path.join(DATA_FOLDER, filename)
            try:
                reader = PdfReader(filepath)
                full_text = ""
                for page in reader.pages:
                    txt = page.extract_text() or ""
                    if txt.strip():
                        full_text += txt + "\n"

                chunk_size = 1400
                overlap = 250
                start, c_num = 0, 0
                while start < len(full_text):
                    end = start + chunk_size
                    chunk = full_text[start:end].strip()
                    if chunk:
                        enriched = f"Institution: ABES Engineering College\nDocument: {filename}\nContent:\n{chunk}"
                        docs.append(enriched)
                        metas.append({"source": filename, "type": "pdf"})
                        ids.append(f"auto_{filename}_{c_num}")
                        c_num += 1
                    start = end - overlap
            except Exception:
                pass

        if docs:
            embeddings = model.encode(docs, show_progress_bar=False).tolist()
            collection.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)

    return model, collection


model, collection = load_system()

api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
if not api_key:
    st.error("🔑 GROQ_API_KEY is not set. Please add it to your Streamlit Cloud Secrets.")
    st.stop()

ai = Groq(api_key=api_key)

# ------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "student_profile" not in st.session_state:
    st.session_state.student_profile = {"branch": None, "year": None}

if "clicked_prompt" not in st.session_state:
    st.session_state.clicked_prompt = None

if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = {
        "active": False,
        "subject": "",
        "questions": [],
        "current_idx": 0,
        "score": 0,
    }

# ------------------------------------------------------------
# Minimal Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 Chat Controls")
    if st.button("➕ New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.student_profile = {"branch": None, "year": None}
        st.session_state.quiz_state = {"active": False, "subject": "", "questions": [], "current_idx": 0, "score": 0}
        st.rerun()

    st.markdown("---")
    branch = st.session_state.student_profile.get("branch") or "Auto"
    year = st.session_state.student_profile.get("year") or "Auto"
    st.caption(f"Profile: **{branch}** • **{year}**")
    st.caption(f"Knowledge Base: **{collection.count()} chunks**")


# ------------------------------------------------------------
# Empty State: Starter Prompt Cards
# ------------------------------------------------------------
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; padding: 15px 0 20px 0;">
        <h3 style="font-weight: 700; margin-bottom: 6px;">How can I help you today?</h3>
        <p style="color: #94a3b8; font-size: 0.92rem;">Select a topic below or type your question in any language:</p>
    </div>
    """, unsafe_allow_html=True)

    starter_cards = [
        ("🛏️ Hostel & Mess Fees", "What is the hostel fee?"),
        ("💰 B.Tech Admission Fees", "What is the B.Tech admission fee?"),
        ("📖 DBMS Syllabus & Units", "What is the DBMS syllabus?"),
        ("📑 Download Question Papers", "previous year question paper"),
    ]

    c1, c2 = st.columns(2)
    for i, (title, prompt_val) in enumerate(starter_cards):
        col = c1 if i % 2 == 0 else c2
        if col.button(title, key=f"starter_card_{i}", use_container_width=True):
            st.session_state.clicked_prompt = prompt_val
            st.rerun()


# ------------------------------------------------------------
# Render Chat History
# ------------------------------------------------------------
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("pyq_files"):
            st.markdown("---")
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
                        key=f"dl_{idx}_{f_idx}_{pdf_file}",
                        use_container_width=True,
                    )


# ------------------------------------------------------------
# In-Chat Interactive Quiz UI
# ------------------------------------------------------------
quiz = st.session_state.quiz_state
if quiz["active"] and quiz["current_idx"] < len(quiz["questions"]):
    curr_q = quiz["questions"][quiz["current_idx"]]
    q_num = quiz["current_idx"] + 1
    total_q = len(quiz["questions"])

    with st.chat_message("assistant"):
        st.markdown(f"**Question {q_num} of {total_q}:** {curr_q['question']}")

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
                feedback = f"✅ **Correct Answer!** ({curr_q['correct_option']})\n\n{curr_q['explanation']}"
                st.session_state.quiz_state["score"] += 1
            else:
                feedback = f"❌ **Incorrect.** You selected ({user_choice}), but the correct answer is **({curr_q['correct_option']})**.\n\n{curr_q['explanation']}"

            st.session_state.messages.append({"role": "user", "content": f"Selected ({user_choice})"})
            st.session_state.messages.append({"role": "assistant", "content": feedback})

            st.session_state.quiz_state["current_idx"] += 1
            if st.session_state.quiz_state["current_idx"] >= total_q:
                final_score = st.session_state.quiz_state["score"]
                summary = f"🎉 **Quiz Finished!** Final Score: **{final_score} / {total_q}**"
                st.session_state.messages.append({"role": "assistant", "content": summary})
                st.session_state.quiz_state["active"] = False
            st.rerun()


# ------------------------------------------------------------
# Input Handling & Stream Pipeline
# ------------------------------------------------------------
user_input = st.chat_input("Message ABES Assistant...")
if st.session_state.clicked_prompt:
    user_input = st.session_state.clicked_prompt
    st.session_state.clicked_prompt = None

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    q_lower = user_input.lower().strip()

    for branch_name in ["cse", "it", "ece", "me", "ee", "civil"]:
        if f" {branch_name} " in f" {q_lower} ":
            st.session_state.student_profile["branch"] = branch_name.upper()
    for year_name in ["1st year", "2nd year", "3rd year", "4th year"]:
        if year_name in q_lower:
            st.session_state.student_profile["year"] = year_name

    # 1. Previous Year Papers
    pyq_triggers = [
        "previous year", "pyq", "pyqs", "question paper", "ques paper", "old question",
        "old paper", "past paper", "exam paper", "purane paper", "sample paper"
    ]
    if any(t in q_lower for t in pyq_triggers):
        folder = get_question_papers_folder()
        all_pdfs = sorted([f for f in os.listdir(folder) if f.lower().endswith(".pdf")]) if os.path.exists(folder) else []

        if all_pdfs:
            stop_words = {"the", "for", "and", "pyq", "pyqs", "ques", "question", "questions", "paper", "papers", "previous", "year", "old", "past", "exam", "give", "me", "show", "i", "want", "of", "in"}
            user_tokens = [w for w in re.findall(r'\b\w+\b', q_lower) if w not in stop_words and len(w) > 2]
            matched = [p for p in all_pdfs if any(t in p.lower() for t in user_tokens)] if user_tokens else []
            selected_pdfs = matched if matched else all_pdfs

            content = f"Here are the previous year question papers available for download (**{len(selected_pdfs)} paper(s)**):"
            st.session_state.messages.append({
                "role": "assistant",
                "content": content,
                "pyq_files": selected_pdfs,
                "pyq_folder": folder
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"No question papers found in `{folder}/`. Please ensure PDF files are placed in that folder."
            })
        st.rerun()

    # 2. Practice Quiz
    quiz_triggers = ["quiz", "test me", "viva question", "ask me question", "mcq"]
    if any(t in q_lower for t in quiz_triggers):
        query_emb = model.encode(user_input).tolist()
        results = collection.query(query_embeddings=[query_emb], n_results=4, include=["documents"])
        docs = results.get("documents", [[]])[0]
        context = "\n".join(docs)

        quiz_prompt = f"""
Create 3 high-quality multiple choice questions based on this subject: {user_input}.
Context:
{context[:2000]}

Respond ONLY with valid JSON array of 3 questions:
[
  {{
    "question": "Question text here?",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct_option": "A",
    "explanation": "Brief explanation why A is correct."
  }}
]
"""
        try:
            res = ai.chat.completions.create(
                model=ACTIVE_MODEL,
                messages=[{"role": "user", "content": quiz_prompt}],
                temperature=0.3,
                max_tokens=1000,
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
                st.session_state.messages.append({"role": "assistant", "content": f"Starting practice quiz on {user_input}."})
                st.rerun()
        except Exception:
            pass

    # 3. Direct Fast Search
    search_query = user_input
    if "hostel" in q_lower or "mess" in q_lower:
        search_query = "hostel fee structure lodging boarding room rent mess charges security deposit AC Non-AC"
    elif "admission" in q_lower or "tuition" in q_lower or "college fee" in q_lower or "btech fee" in q_lower or "fee" in q_lower:
        search_query = "academic fee tuition fee admission schedule of charges b.tech"
    elif "syllabus" in q_lower:
        search_query = user_input + " complete subject syllabus units course code detailed"
    elif "exam" in q_lower or "schedule" in q_lower:
        search_query = "academic calendar examination schedule sessional test end sem"
    elif "placement" in q_lower or "package" in q_lower:
        search_query = "placement statistics packages highest average companies recruitment"

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
            documents.append(doc)
            if meta and "source" in meta:
                sources.add(meta["source"])
    except Exception:
        pass

    # 4. Streamed Generation
    exam_triggers = ["exam", "exams", "schedule", "date sheet", "datesheet", "when are exams", "exam date"]
    is_exam_query = any(trigger in q_lower for trigger in exam_triggers)
    is_placement_query = any(trigger in q_lower for trigger in ["placement", "package", "packages", "salary", "companies", "recruiters"])

    with st.chat_message("assistant"):
        if documents or is_exam_query or is_placement_query:
            context = "\n\n---\n\n".join(documents)
            student_ctx = f"Branch: {st.session_state.student_profile['branch'] or 'General'}, Year: {st.session_state.student_profile['year'] or 'All'}"

            system_prompt = f"""
You are the official AI Assistant for ABES Engineering College (AKTU).
Tone: Conversational, helpful, clear, and student-friendly.
Student Context: {student_ctx}

INSTRUCTIONS:
1. LANGUAGE: Reply in the same language as the student (English, Hindi, or Hinglish).
2. HOSTEL FEES: Provide complete fee breakdowns (Non-AC vs AC, 2-seater, 3-seater, 4-seater, laundry, security deposit) using clean Markdown tables.
3. ADMISSION FEES: Present itemized tuition and academic fees using clean Markdown tables.
4. EXAM QUERIES: If official dates are in documents, display them in a table. Otherwise explain the standard AKTU cycle (Odd sem: Dec–Jan, Even sem: May–June).
5. PLACEMENT QUERIES: Explain that students should visit the Training & Placement Cell (T&P) or abes.ac.in for official placement statistics, mentioning top recruiters like TCS, Infosys, Wipro, Capgemini, and Cognizant.
6. FORMATTING: Use Markdown tables and bullet points. End with a helpful follow-up question.
"""

            messages = [{"role": "system", "content": system_prompt}]
            for msg in st.session_state.messages[-4:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({
                "role": "user",
                "content": f"University Documents:\n{context}\n\nStudent Question: {user_input}",
            })

            try:
                stream = ai.chat.completions.create(
                    model=ACTIVE_MODEL,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2500,
                    stream=True,
                )

                def chunk_generator():
                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content

                full_answer = st.write_stream(chunk_generator())

                if sources:
                    source_tags = "".join(f"<span class='source-tag'>📄 {s}</span> " for s in sorted(sources))
                    st.markdown(f"<div style='margin-top: 14px;'><b>Sources:</b><br>{source_tags}</div>", unsafe_allow_html=True)
                    full_answer += "\n\n**Sources:** " + ", ".join(f"`{s}`" for s in sorted(sources))

            except Exception:
                res = ai.chat.completions.create(
                    model=ACTIVE_MODEL,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2500,
                )
                full_answer = res.choices[0].message.content
                st.markdown(full_answer)

        else:
            full_answer = "This information does not appear to be present in the available university documents. Please check with the university office or the official website (abes.ac.in)."
            st.markdown(full_answer)

    st.session_state.messages.append({"role": "assistant", "content": full_answer})

st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.75rem; margin-top: 24px;'>ABES Assistant can make mistakes. Verify critical dates and fees with the college office.</p>", unsafe_allow_html=True)