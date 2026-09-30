import json
import os
import re
import time

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
import streamlit as st

st.set_page_config(
    page_title="ABES University Assistant",
    page_icon="🎓",
    layout="centered",
)

LOGO_FILE = "university_logo.png"

# Automatically find question papers folder
def get_question_papers_folder():
    candidates = ["question_papers", "question_paper", "questionpapers", "pyq", "pyqs", "data/question_papers"]
    for folder in candidates:
        if os.path.exists(folder) and os.path.isdir(folder):
            return folder
    return "question_papers"

QUESTION_PAPERS_FOLDER = get_question_papers_folder()

# ------------------------------------------------------------
# Header with logo
# ------------------------------------------------------------
col1, col2 = st.columns([1.2, 5])
with col1:
    if os.path.exists(LOGO_FILE):
        st.image(LOGO_FILE, width=120)
with col2:
    st.markdown("## 🎓 ABES University Assistant")
    st.caption("Official AI helper for students, powered by university documents")


@st.cache_resource
def load_system():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="db")
    collection = client.get_or_create_collection(name="university")
    return model, collection


model, collection = load_system()

api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
if not api_key:
    st.error("GROQ_API_KEY is not set. Please set it in environment or Streamlit secrets.")
    st.stop()

ai = Groq(api_key=api_key)


@st.cache_resource
def get_working_groq_model():
    candidates = ["openai/gpt-oss-20b", "llama-3.1-8b-instant", "llama3-8b-8192", "mixtral-8x7b-32768"]
    try:
        available_ids = [m.id for m in ai.models.list().data]
        for c in candidates:
            if c in available_ids:
                return c
        if available_ids:
            return available_ids[0]
    except Exception:
        pass
    return "openai/gpt-oss-20b"


ACTIVE_MODEL = get_working_groq_model()

# ------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "student_profile" not in st.session_state:
    st.session_state.student_profile = {"branch": None, "year": None}

if "suggestions" not in st.session_state:
    st.session_state.suggestions = [
        "What is the hostel fee?",
        "What is the admission fee for B.Tech?",
        "What is the DBMS syllabus?",
        "Previous year question papers",
    ]

if "clicked_suggestion" not in st.session_state:
    st.session_state.clicked_suggestion = None

if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = {
        "active": False,
        "subject": "",
        "questions": [],
        "current_idx": 0,
        "score": 0,
    }

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.header("🎓 University AI")
    if st.button("🗑️ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.student_profile = {"branch": None, "year": None}
        st.session_state.quiz_state = {"active": False, "subject": "", "questions": [], "current_idx": 0, "score": 0}
        st.session_state.suggestions = [
            "What is the hostel fee?",
            "What is the admission fee for B.Tech?",
            "What is the DBMS syllabus?",
            "Previous year question papers",
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("### 🧑‍🎓 Student Memory")
    branch = st.session_state.student_profile.get("branch") or "Not set"
    year = st.session_state.student_profile.get("year") or "Not set"
    st.caption(f"**Branch:** {branch} | **Year:** {year}")

    st.markdown("---")
    st.write("Topics covered:")
    st.write("• Hostel & Mess Fees (AC / Non-AC)\n• Admission & Academic Fees\n• Syllabus & Exam Dates\n• Interactive Practice Quizzes\n• Previous Year Question Papers")


# ------------------------------------------------------------
# Smart Search Query Generator (Separates Hostel from Admission)
# ------------------------------------------------------------
def get_search_queries(user_query: str) -> list[str]:
    """Generates precise search terms ensuring hostel and admission queries never collide."""
    q_lower = user_query.lower()
    queries = [user_query]

    # Clean out institution acronym noise if present
    clean = re.sub(r"\babes\b", "", q_lower, flags=re.IGNORECASE).strip()
    if clean and clean != q_lower:
        queries.append(clean)

    # Specific category routing:
    if "hostel" in q_lower or "mess" in q_lower:
        queries.append("hostel fee structure lodging boarding mess charges security deposit AC Non-AC seater")
    elif "admission" in q_lower or "tuition" in q_lower or "btech fee" in q_lower or "college fee" in q_lower or "abes fee" in q_lower:
        queries.append("academic fee tuition fee admission schedule of charges b.tech")
    elif "syllabus" in q_lower or "unit" in q_lower:
        queries.append(user_query + " complete syllabus units topics course code")
    elif "exam" in q_lower or "schedule" in q_lower or "date" in q_lower:
        queries.append("academic calendar examination schedule sessional sessional test end sem")
    elif "placement" in q_lower or "package" in q_lower:
        queries.append("placement statistics packages highest average companies recruitment")

    return queries


# ------------------------------------------------------------
# Interactive Quiz Generator
# ------------------------------------------------------------
def generate_interactive_quiz(subject_text: str, context: str) -> list[dict]:
    prompt = f"""
Create 3 high-quality multiple choice questions based on this university subject: {subject_text}.
Context:
{context[:2000]}

Respond ONLY with valid JSON array containing exactly 3 questions:
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
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        text = res.choices[0].message.content.strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return []


# ------------------------------------------------------------
# Render Chat History (With Permanent Download Buttons)
# ------------------------------------------------------------
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Welcome! I'm your ABES University assistant. "
            "I can help you with hostel fees, admission charges, syllabus, exams, or question papers! What would you like to know?"
        )

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
                        label=f"📥 Download {subject_name} (PDF)",
                        data=file_bytes,
                        file_name=pdf_file,
                        mime="application/pdf",
                        key=f"dl_{idx}_{f_idx}_{pdf_file}",
                        use_container_width=True,
                    )

# ------------------------------------------------------------
# Active Quiz UI
# ------------------------------------------------------------
quiz = st.session_state.quiz_state
if quiz["active"] and quiz["current_idx"] < len(quiz["questions"]):
    curr_q = quiz["questions"][quiz["current_idx"]]
    q_num = quiz["current_idx"] + 1
    total_q = len(quiz["questions"])

    with st.chat_message("assistant"):
        st.markdown(f"### 🎯 Quiz: {quiz['subject']} (Question {q_num}/{total_q})")
        st.markdown(f"**{curr_q['question']}**")

        c1, c2 = st.columns(2)
        cols = [c1, c2, c1, c2]

        user_choice = None
        for i, opt in enumerate(curr_q["options"]):
            letter = opt.split(")")[0].strip()
            if cols[i].button(opt, key=f"quiz_opt_{q_num}_{i}", use_container_width=True):
                user_choice = letter

        if st.button("❌ Quit Quiz", key="quit_quiz_btn"):
            st.session_state.quiz_state["active"] = False
            st.rerun()

        if user_choice:
            is_correct = user_choice.upper() == curr_q["correct_option"].upper()
            if is_correct:
                feedback = f"✅ **Correct!** ({curr_q['correct_option']})\n\n{curr_q['explanation']}"
                st.session_state.quiz_state["score"] += 1
            else:
                feedback = f"❌ **Incorrect!** You chose ({user_choice}), but the correct answer is **({curr_q['correct_option']})**.\n\n{curr_q['explanation']}"

            st.session_state.messages.append({"role": "user", "content": f"My answer: {user_choice}"})
            st.session_state.messages.append({"role": "assistant", "content": feedback})

            st.session_state.quiz_state["current_idx"] += 1
            if st.session_state.quiz_state["current_idx"] >= total_q:
                final_score = st.session_state.quiz_state["score"]
                summary = f"🎉 **Quiz Finished!**\n\nYour Final Score: **{final_score} / {total_q}**"
                st.session_state.messages.append({"role": "assistant", "content": summary})
                st.session_state.quiz_state["active"] = False
            st.rerun()

# ------------------------------------------------------------
# Quick Suggestions Buttons
# ------------------------------------------------------------
valid_suggestions = [s for s in st.session_state.suggestions if s and len(s.strip()) > 3]
if valid_suggestions and not quiz["active"]:
    st.write("💡 **Quick suggestions:**")
    num_cols = min(len(valid_suggestions), 3)
    sug_cols = st.columns(num_cols)
    for i, sug_text in enumerate(valid_suggestions[:num_cols]):
        if sug_cols[i].button(sug_text, key=f"chip_{i}_{len(st.session_state.messages)}", use_container_width=True):
            st.session_state.clicked_suggestion = sug_text
            st.rerun()

# ------------------------------------------------------------
# Input Handling
# ------------------------------------------------------------
user_input = st.chat_input("Ask anything about your university...")
if st.session_state.clicked_suggestion:
    user_input = st.session_state.clicked_suggestion
    st.session_state.clicked_suggestion = None

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    question_lower = user_input.lower().strip()

    # Track student profile if mentioned
    for branch_name in ["cse", "it", "ece", "me", "ee", "civil"]:
        if f" {branch_name} " in f" {question_lower} ":
            st.session_state.student_profile["branch"] = branch_name.upper()
    for year_name in ["1st year", "2nd year", "3rd year", "4th year"]:
        if year_name in question_lower:
            st.session_state.student_profile["year"] = year_name

    # --------------------------------------------------------
    # 1. Previous Year Question Papers (PYQ)
    # --------------------------------------------------------
    pyq_triggers = [
        "previous year", "pyq", "pyqs", "question paper", "ques paper", "old question",
        "old paper", "past paper", "exam paper", "purane paper", "sample paper"
    ]
    if any(t in question_lower for t in pyq_triggers):
        folder = get_question_papers_folder()
        all_pdfs = sorted([f for f in os.listdir(folder) if f.lower().endswith(".pdf")]) if os.path.exists(folder) else []

        if all_pdfs:
            stop_words = {"the", "for", "and", "pyq", "pyqs", "ques", "question", "questions", "paper", "papers", "previous", "year", "old", "past", "exam", "give", "me", "show", "i", "want", "of", "in"}
            user_tokens = [w for w in re.findall(r'\b\w+\b', question_lower) if w not in stop_words and len(w) > 2]
            matched = [p for p in all_pdfs if any(t in p.lower() for t in user_tokens)] if user_tokens else []
            selected_pdfs = matched if matched else all_pdfs

            content = f"Here are the previous year question papers available for download (**{len(selected_pdfs)} paper(s)**). Click below to download:"
            st.session_state.messages.append({
                "role": "assistant",
                "content": content,
                "pyq_files": selected_pdfs,
                "pyq_folder": folder
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ No PDF files found in the `{folder}` folder. Please place your question paper PDFs in that directory."
            })

        st.session_state.suggestions = ["What is the hostel fee?", "What is the DBMS syllabus?", "Take a practice quiz"]
        st.rerun()

    # --------------------------------------------------------
    # 2. Check for Quiz Request
    # --------------------------------------------------------
    quiz_triggers = ["quiz", "test me", "viva question", "ask me question", "mcq"]
    if any(t in question_lower for t in quiz_triggers):
        query_embedding = model.encode(user_input).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=4, include=["documents"])
        docs = results.get("documents", [[]])[0]
        context = "\n".join(docs)

        questions = generate_interactive_quiz(user_input, context)
        if questions:
            st.session_state.quiz_state = {
                "active": True,
                "subject": user_input.title(),
                "questions": questions,
                "current_idx": 0,
                "score": 0,
            }
            with st.chat_message("assistant"):
                st.markdown(f"🚀 Starting interactive quiz on **{user_input.title()}**! Click your answers below:")
            st.session_state.messages.append({"role": "assistant", "content": f"Starting quiz on {user_input}."})
            st.rerun()

    # --------------------------------------------------------
    # 3. Dual-Query RAG Search (Guarantees Hostel & Fee Accuracy)
    # --------------------------------------------------------
    with st.status("🔍 Searching university records...", expanded=False) as status:
        search_terms = get_search_queries(user_input)
        documents = []
        sources = set()
        seen_chunks = set()

        try:
            for term in search_terms:
                emb = model.encode(term).tolist()
                results = collection.query(
                    query_embeddings=[emb],
                    n_results=5,
                    include=["documents", "metadatas", "distances"],
                )
                raw_docs = results.get("documents", [[]])[0]
                raw_meta = results.get("metadatas", [[]])[0]

                for doc, meta in zip(raw_docs, raw_meta):
                    short_key = doc[:120].strip()
                    if short_key not in seen_chunks:
                        seen_chunks.add(short_key)
                        documents.append(doc)
                        if meta and "source" in meta:
                            sources.add(meta["source"])

        except Exception as e:
            st.error(f"Search error: {e}")

        status.update(label="🤖 Generating answer...", state="running")

    # --------------------------------------------------------
    # Response Generation
    # --------------------------------------------------------
    exam_triggers = ["exam", "exams", "schedule", "date sheet", "datesheet", "when are exams", "exam date"]
    is_exam_query = any(trigger in question_lower for trigger in exam_triggers)
    is_placement_query = any(trigger in question_lower for trigger in ["placement", "package", "packages", "salary", "companies", "recruiters"])

    with st.chat_message("assistant"):
        if documents or is_exam_query or is_placement_query:
            context = "\n\n---\n\n".join(documents)

            student_ctx = f"Branch: {st.session_state.student_profile['branch'] or 'General'}, Year: {st.session_state.student_profile['year'] or 'All'}"

            system_prompt = f"""
You are the official AI Assistant for ABES Engineering College (AKTU).
Tone: Clear, polite, student-friendly, and informative.
Student Context: {student_ctx}

INSTRUCTIONS:
1. LANGUAGE:
   - Reply in the same language style as the student (English, Hindi, or Hinglish).
2. HOSTEL FEES:
   - When asked about hostel fees, provide the complete fee breakdown (Non-AC vs AC, 2-seater, 3-seater, 4-seater, laundry, security deposit) using clean Markdown tables.
3. EXAM QUERIES:
   - If official dates are in documents, show them. Otherwise explain the standard AKTU schedule (Odd sem: Dec–Jan, Even sem: May–June) and suggest checking the ABES student notice board.
4. PLACEMENT QUERIES:
   - If official placement figures are in the documents, display them clearly.
   - If specific placement package statistics are not in the uploaded documents, explain that students should visit the Training & Placement Cell (T&P) or abes.ac.in for the official placement report, and mention that top recruiters include companies like TCS, Infosys, Wipro, Cognizant, and Capgemini.
5. FORMATTING: Use Markdown tables for numbers and bullet points for lists. Always end with 1 helpful follow-up question.
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
                    max_tokens=1500,
                    stream=True,
                )

                def chunk_generator():
                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content

                full_answer = st.write_stream(chunk_generator())

                if sources:
                    source_box = "\n\n---\n**Sources:**\n" + "\n".join(f"- 📄 `{s}`" for s in sorted(sources))
                    st.markdown(source_box)
                    full_answer += source_box

            except Exception:
                res = ai.chat.completions.create(
                    model=ACTIVE_MODEL,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1500,
                )
                full_answer = res.choices[0].message.content
                st.markdown(full_answer)

        else:
            full_answer = "This information does not appear to be present in the available university documents. Please check with the university administrative office or the official website (abes.ac.in)."
            st.markdown(full_answer)

    # Dynamic suggestions for next turn
    if "hostel" in question_lower:
        st.session_state.suggestions = ["What are the mess facilities?", "What is the B.Tech admission fee?", "Previous year question papers"]
    elif "fee" in question_lower:
        st.session_state.suggestions = ["What is the hostel fee?", "Are there scholarships available?", "Previous year question papers"]
    elif "exam" in question_lower:
        st.session_state.suggestions = ["What is the DBMS syllabus?", "Take a practice quiz", "Previous year question papers"]
    else:
        st.session_state.suggestions = ["What is the hostel fee?", "What is the admission fee?", "Previous year question papers"]

    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    st.rerun()

st.markdown("---")
st.caption("⚠️ ABES Assistant provides guidance based on official documents. For official confirmations, verify with the administration.")