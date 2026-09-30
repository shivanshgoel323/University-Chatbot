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

# Automatically find the question papers folder
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
        "What is the admission fee for B.Tech?",
        "What is the hostel fee?",
        "What is the exam schedule?",
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
            "What is the admission fee for B.Tech?",
            "What is the hostel fee?",
            "What is the exam schedule?",
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
    st.write("• Fees & Scholarships\n• Hostel & Mess Charges\n• Exam Dates & Schedules\n• Syllabus & Course Details\n• Interactive Practice Quizzes\n• Previous Year Question Papers")


# ------------------------------------------------------------
# Query Analyzer
# ------------------------------------------------------------
def analyze_query(user_query: str, history: list, profile: dict):
    recent_context = ""
    if history:
        recent_context = "\n".join(f"{m['role']}: {m['content']}" for m in history[-2:])

    system_prompt = f"""
You are an NLP analyzer for an ABES University Chatbot.
Known Student: Branch: {profile.get('branch')}, Year: {profile.get('year')}

Tasks:
1. Detect user's language: 'English', 'Hindi', or 'Hinglish'.
2. Extract branch (CSE, IT, ECE, ME, etc.) or year (1st, 2nd, 3rd, 4th) if mentioned.
3. Optimize the query into 2-5 English keywords for searching university documents:
   - If user asks about 'exams', rewrite to: 'academic calendar examination schedule sessional end sem theory practical datesheet'.
   - If user asks about 'abes fee', rewrite to: 'admission fee tuition fee structure schedule of charges'.
   - If user asks about 'hostel', rewrite to: 'hostel fee room rent mess charges security'.
4. Provide 3 short, relevant follow-up questions the student might ask next.

Respond ONLY in valid JSON:
{{
  "detected_language": "<English|Hindi|Hinglish>",
  "detected_branch": "<CSE|IT|ECE|null>",
  "detected_year": "<1st|2nd|3rd|4th|null>",
  "search_query": "<english search keywords>",
  "follow_ups": ["<suggestion 1>", "<suggestion 2>", "<suggestion 3>"]
}}
"""
    try:
        response = ai.chat.completions.create(
            model=ACTIVE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{recent_context}\n\nMessage: {user_query}"},
            ],
            temperature=0.0,
            max_tokens=220,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if data.get("detected_branch"):
                profile["branch"] = data["detected_branch"]
            if data.get("detected_year"):
                profile["year"] = data["detected_year"]

            search_query = data.get("search_query", user_query)
            lang = data.get("detected_language", "English")
            follow_ups = [s.strip() for s in data.get("follow_ups", []) if s.strip()]
            return search_query, lang, follow_ups
    except Exception:
        pass

    clean = re.sub(r"\babes\b", "", user_query, flags=re.IGNORECASE).strip()
    if "fee" in clean.lower():
        clean += " admission fee tuition fee structure"
    return clean or user_query, "English", []


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
            "I can help you with admission fees, hostel charges, syllabus, exam dates, placements, or question papers! What would you like to know?"
        )

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Render download buttons permanently for question paper messages
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

    # --------------------------------------------------------
    # 1. Previous Year Question Papers (PYQ)
    # --------------------------------------------------------
    pyq_triggers = [
        "previous year", "pyq", "pyqs", "question paper", "ques paper", "old question",
        "old paper", "past paper", "exam paper", "purane paper", "sample paper"
    ]
    if any(t in question_lower for t in pyq_triggers):
        folder = get_question_papers_folder()
        if os.path.exists(folder):
            all_pdfs = sorted([f for f in os.listdir(folder) if f.lower().endswith(".pdf")])
        else:
            all_pdfs = []

        if all_pdfs:
            # Check if user mentioned a specific subject (e.g. "DBMS pyq")
            stop_words = {"the", "for", "and", "pyq", "pyqs", "ques", "question", "questions", "paper", "papers", "previous", "year", "old", "past", "exam", "give", "me", "show", "i", "want", "of", "in"}
            user_tokens = [w for w in re.findall(r'\b\w+\b', question_lower) if w not in stop_words and len(w) > 2]

            matched = []
            if user_tokens:
                for p in all_pdfs:
                    if any(t in p.lower() for t in user_tokens):
                        matched.append(p)

            selected_pdfs = matched if matched else all_pdfs

            if matched:
                content = f"Here are the previous year question papers matching your request (**{len(matched)} paper(s) found**). Click below to download:"
            else:
                content = f"Here are all available previous year question papers (**{len(all_pdfs)} paper(s) found**). Click below to download:"

            st.session_state.messages.append({
                "role": "assistant",
                "content": content,
                "pyq_files": selected_pdfs,
                "pyq_folder": folder
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ No PDF files found in the `{folder}` folder. Please place your question paper PDFs in that folder."
            })

        st.session_state.suggestions = ["What is the fee structure?", "What is the exam schedule?", "Take a practice quiz"]
        st.rerun()

    # --------------------------------------------------------
    # 2. Check for Quiz Request
    # --------------------------------------------------------
    quiz_triggers = ["quiz", "test me", "viva question", "ask me question", "mcq"]
    if any(t in question_lower for t in quiz_triggers):
        search_query, _, _ = analyze_query(user_input, st.session_state.messages, st.session_state.student_profile)
        query_embedding = model.encode(search_query).tolist()
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
    # 3. Standard RAG Retrieval & Generation
    # --------------------------------------------------------
    with st.status("🔍 Searching university records...", expanded=False) as status:
        search_query, detected_lang, follow_ups = analyze_query(
            user_input, st.session_state.messages, st.session_state.student_profile
        )

        documents = []
        sources = set()

        try:
            query_embedding = model.encode(search_query).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=6,
                include=["documents", "metadatas", "distances"],
            )

            raw_docs = results.get("documents", [[]])[0]
            raw_meta = results.get("metadatas", [[]])[0]
            raw_dist = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(raw_docs, raw_meta, raw_dist):
                if dist < 2.0:
                    documents.append(doc)
                    source_name = meta.get("source", "University Document") if meta else "University Document"
                    sources.add(source_name)

        except Exception as e:
            st.error(f"Search error: {e}")

        status.update(label="🤖 Generating answer...", state="running")

    exam_triggers = ["exam", "exams", "schedule", "date sheet", "datesheet", "when are exams", "exam date"]
    is_exam_query = any(trigger in question_lower for trigger in exam_triggers)

    with st.chat_message("assistant"):
        if documents or is_exam_query:
            context = "\n\n---\n\n".join(documents)

            student_ctx = f"Branch: {st.session_state.student_profile['branch'] or 'General'}, Year: {st.session_state.student_profile['year'] or 'All'}"

            system_prompt = f"""
You are the official AI Assistant for ABES Engineering College (AKTU).
Tone: Clear, polite, student-friendly, and informative.
Student Context: {student_ctx}

INSTRUCTIONS:
1. LANGUAGE: Reply in the same language as the student ({detected_lang} - English, Hindi, or Hinglish).
2. EXAM QUERIES:
   - If official dates are in the documents, show them in a Markdown schedule table (CT-1, CT-2, PUT, End-Sem Theory & Practical).
   - If specific branch dates are pending, explain the standard AKTU schedule:
     * Odd Semesters (1st, 3rd, 5th, 7th): Internal Sessionals in Oct/Nov, End-Sem Theory Exams in December–January.
     * Even Semesters (2nd, 4th, 6th, 8th): Internal Sessionals in March/April, End-Sem Theory Exams in May–June.
     * Advise checking the ABES student notice board or AKTU ERP portal, and ask which semester they are preparing for.
3. GROUNDING: Use retrieved documents as truth. Never invent fee amounts.
4. FORMATTING: Use Markdown tables for fee structures and bullet points for lists.
5. PROACTIVE CLOSING: End with 1 helpful follow-up question.
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
            if detected_lang == "Hindi":
                full_answer = "माफ़ कीजिये, यह जानकारी विश्वविद्यालय के दस्तावेज़ों में नहीं मिली। कृपया कॉलेज कार्यालय या abes.ac.in पर संपर्क करें।"
            elif detected_lang == "Hinglish":
                full_answer = "Yeh information abhi uploaded documents mein nahi mili hai. Please college office ya abes.ac.in check karein."
            else:
                full_answer = "This information does not appear to be present in the available university documents. Please check with the university office or official website (abes.ac.in)."
            st.markdown(full_answer)

    if follow_ups and any(len(f.strip()) > 3 for f in follow_ups):
        st.session_state.suggestions = [f for f in follow_ups if len(f.strip()) > 3]
    else:
        st.session_state.suggestions = [
            "What is the hostel fee?",
            "Take a quiz on this topic",
            "What are the placement packages?",
        ]

    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    st.rerun()

st.markdown("---")
st.caption("⚠️ ABES Assistant provides guidance based on official documents. For official confirmations, verify with the administration.")