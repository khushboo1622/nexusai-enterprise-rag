"""
frontend/pages/chat.py - NexusAI Chat Page
Full width chat, clean HTML top bar, markdown rendering
"""

import streamlit as st
import requests
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

API_BASE = os.getenv("API_BASE_URL", "http://localhost:5050")

st.set_page_config(
    page_title="NexusAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if not st.session_state.get("logged_in"):
    st.switch_page("app.py")

st.markdown("""
<style>
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    [data-testid="stSidebarNav"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    .block-container {
        padding-top: 0 !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 820px !important;
        margin: 0 auto !important;
    }

    /* Top bar - pure HTML, no Streamlit columns */
    .top-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0 0.8rem 0;
        border-bottom: 1px solid #23253A;
        margin-bottom: 1.8rem;
    }
    .top-bar-logo {
        font-size: 1.15rem;
        font-weight: 700;
        color: #6C63FF;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .top-bar-right {
        display: flex;
        align-items: center;
        gap: 0.9rem;
    }
    .top-user-name {
        font-size: 0.88rem;
        font-weight: 600;
        color: #DADBE8;
    }
    .role-pill {
        border-radius: 20px;
        padding: 3px 11px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.7px;
        text-transform: uppercase;
        background: #23253A;
    }
    .logout-btn {
        font-size: 0.78rem;
        color: #6B6D85;
        cursor: pointer;
        border: 1px solid #2D2F3E;
        border-radius: 20px;
        padding: 3px 13px;
        text-decoration: none;
        transition: all 0.2s;
        background: transparent;
    }
    .logout-btn:hover { color: #FF4B4B; border-color: #FF4B4B; }

    /* Greeting */
    .greeting {
        font-size: 1.55rem;
        font-weight: 700;
        color: #EEEEF5;
        margin-bottom: 0.15rem;
    }
    .greeting-sub {
        font-size: 0.86rem;
        color: #5C5E78;
        margin-bottom: 1.4rem;
    }

    /* Welcome box */
    .welcome-box {
        background: #13151F;
        border: 1px solid #23253A;
        border-left: 3px solid #6C63FF;
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1.4rem;
        font-size: 0.85rem;
        color: #7B7D98;
        line-height: 1.6;
    }

    /* Chat bubbles */
    .user-bubble {
        background: #5C55E8;
        color: #fff;
        border-radius: 18px 18px 4px 18px;
        padding: 0.7rem 1.1rem;
        margin: 0.45rem 0 0.45rem 18%;
        font-size: 0.93rem;
        line-height: 1.6;
        word-wrap: break-word;
    }
    .assistant-bubble {
        background: #16182A;
        color: #D8D9E8;
        border-radius: 18px 18px 18px 4px;
        padding: 0.85rem 1.15rem;
        margin: 0.45rem 18% 0.45rem 0;
        border: 1px solid #23253A;
        font-size: 0.93rem;
        line-height: 1.75;
        word-wrap: break-word;
    }
    .assistant-bubble strong { color: #9D97F5; font-weight: 600; }
    .assistant-bubble ul { margin: 0.35rem 0 0.35rem 1.1rem; padding: 0; }
    .assistant-bubble li { margin-bottom: 0.28rem; }
    .assistant-bubble p { margin: 0.25rem 0; }

    /* Feedback buttons */
    .feedback-row {
        display: flex;
        gap: 0.4rem;
        margin-top: 0.3rem;
        margin-left: 0.2rem;
    }
    .feedback-btn {
        background: transparent;
        border: 1px solid #2D2F3E;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        cursor: pointer;
        color: #6B6D85;
        transition: all 0.15s;
    }
    .feedback-btn:hover { border-color: #6C63FF; color: #6C63FF; }
    .feedback-given { color: #10B981; font-size: 0.75rem; margin-left: 0.3rem; }

    /* Streamlit logout button hidden - we use HTML button */
    div[data-testid="stButton"] > button {
        background: transparent !important;
        color: #6B6D85 !important;
        border: 1px solid #2D2F3E !important;
        border-radius: 20px !important;
        padding: 0.18rem 0.85rem !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        min-height: 0 !important;
        height: auto !important;
        line-height: 1.4 !important;
    }
    div[data-testid="stButton"] > button:hover {
        color: #FF4B4B !important;
        border-color: #FF4B4B !important;
        background: transparent !important;
    }

    /* Chat input */
    div[data-testid="stChatInput"] textarea {
        background: #16182A !important;
        border: 1px solid #23253A !important;
        border-radius: 12px !important;
        color: #EEEEF5 !important;
        font-size: 0.93rem !important;
    }
    div[data-testid="stChatInput"] {
        border-top: 1px solid #23253A;
        padding-top: 0.6rem;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────
def get_first_name(full_name: str) -> str:
    return full_name.strip().split()[0] if full_name.strip() else full_name


def get_role_color(role: str) -> str:
    return {
        "HR":          "#10B981",
        "FINANCE":     "#F59E0B",
        "ENGINEERING": "#3B82F6",
        "MARKETING":   "#EC4899",
        "C_LEVEL":     "#6C63FF",
    }.get(role, "#6C63FF")


def format_answer(text: str) -> str:
    """Convert LLM markdown to clean HTML."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)

    lines = text.split('\n')
    html_lines = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('* ') or stripped.startswith('- '):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            html_lines.append(f'<li>{stripped[2:]}</li>')
        else:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            if stripped:
                html_lines.append(f'<p>{stripped}</p>')

    if in_list:
        html_lines.append('</ul>')

    return ''.join(html_lines)


def call_chat_api(question: str) -> tuple[str, str | None]:
    """Returns (answer, log_id). log_id used for feedback."""
    try:
        response = requests.post(
            f"{API_BASE}/chat/query",
            json={"question": question},
            headers={"Authorization": f"Bearer {st.session_state['token']}"},
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            return data["answer"], data.get("log_id")
        elif response.status_code == 401:
            st.session_state["logged_in"] = False
            st.switch_page("app.py")
        else:
            return "Sorry, something went wrong. Please try again.", None
    except requests.exceptions.Timeout:
        return "The request took too long. Please try again.", None
    except requests.exceptions.ConnectionError:
        return "Cannot reach the server. Please check your connection.", None


def submit_feedback(log_id: str, feedback: str):
    """Submit thumbs up/down feedback to backend."""
    try:
        requests.patch(
            f"{API_BASE}/chat/feedback/{log_id}",
            params={"feedback": feedback},
            headers={"Authorization": f"Bearer {st.session_state['token']}"},
            timeout=5,
        )
    except Exception:
        pass  # feedback failure should never affect UX


def call_chat_api_streaming(prompt: str):
    """
    Stream tokens from backend using SSE (Server-Sent Events).
    Yields tokens one by one as they arrive from Groq.
    Returns (full_answer, log_id) when done.

    Why streaming feels faster:
    - First token appears in ~0.3s instead of waiting 3-4s for full answer
    - User sees the AI "thinking" in real time — much better UX
    """
    import json

    headers = {
        "Authorization": f"Bearer {st.session_state['token']}",
        "Accept": "text/event-stream",
    }

    full_answer = ""
    log_id = None

    try:
        with requests.post(
            f"{API_BASE}/chat/stream",
            json={"question": prompt},
            headers=headers,
            stream=True,
            timeout=60,
        ) as response:
            if response.status_code != 200:
                yield "Sorry, something went wrong. Please try again."
                return

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "token" in data:
                            token = data["token"]
                            full_answer += token
                            yield token
                        elif data.get("done"):
                            log_id = data.get("log_id")
                            break

    except requests.exceptions.Timeout:
        yield "The request took too long. Please try again."
    except requests.exceptions.ConnectionError:
        yield "Cannot reach the server. Please check your connection."

    # Store result in session after streaming completes
    st.session_state["_last_stream_result"] = {
        "answer": full_answer,
        "log_id": log_id,
    }


def do_logout():
    try:
        requests.delete(
            f"{API_BASE}/chat/session",
            headers={"Authorization": f"Bearer {st.session_state.get('token', '')}"},
            timeout=5,
        )
    except Exception:
        pass
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.switch_page("app.py")


# ── User info ──────────────────────────────────────────────────────────────
user_name  = st.session_state.get("user_name", "User")
user_role  = st.session_state.get("user_role", "")
first_name = get_first_name(user_name)
role_color = get_role_color(user_role)
# Show actual department to user, not the mapped role
# Role is used internally for RBAC — department is what user knows themselves as
user_department = st.session_state.get("department", user_role)
role_label = user_department.upper() if user_department else user_role.replace("_", " ")

# ── Top bar — pure HTML so layout never breaks ─────────────────────────────
st.markdown(f"""
<div class="top-bar">
    <div class="top-bar-logo">🧠 NexusAI</div>
    <div class="top-bar-right">
        <span class="top-user-name">{user_name}</span>
        <span class="role-pill" style="color:{role_color};">{role_label}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Logout button — Streamlit button placed after HTML bar
# Aligned right using columns
_, col_logout = st.columns([9, 1])
with col_logout:
    if st.button("Sign out"):
        do_logout()

st.markdown('<div style="margin-top:-2.8rem;"></div>', unsafe_allow_html=True)

# ── Greeting ───────────────────────────────────────────────────────────────
st.markdown(f'<div class="greeting">Hi, {first_name} 👋</div>', unsafe_allow_html=True)
st.markdown('<div class="greeting-sub">Ask me anything about your company documents.</div>', unsafe_allow_html=True)

# Welcome box on fresh session
if not st.session_state.get("messages"):
    access_map = {
        "HR":          "HR policies, employee records and general company info",
        "FINANCE":     "financial reports, budgets and general company info",
        "ENGINEERING": "technical docs, API guidelines and general company info",
        "MARKETING":   "marketing strategies, campaigns and general company info",
        "C_LEVEL":     "all company documents across every department",
    }
    st.markdown(f"""
    <div class="welcome-box">
        You have access to <strong style="color:#9D97F5;">
        {access_map.get(user_role, "company documents")}</strong>.
        We do not store your questions, answers, or identity.
        Only anonymous feedback (👍/👎) and guardrail block reasons are logged.
        Chat history exists only in this session until you sign out.
    </div>
    """, unsafe_allow_html=True)

# ── Chat history ───────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.get("messages", [])):
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="assistant-bubble">{format_answer(msg["content"])}</div>',
            unsafe_allow_html=True,
        )
        # Show feedback buttons for assistant messages that have a log_id
        log_id = msg.get("log_id")
        if log_id:
            feedback_given = msg.get("feedback")
            if feedback_given:
                st.markdown(
                    f'<span class="feedback-given">{"👍" if feedback_given == "positive" else "👎"} Feedback recorded</span>',
                    unsafe_allow_html=True,
                )
            else:
                col1, col2, col3 = st.columns([0.06, 0.06, 10])
                with col1:
                    if st.button("👍", key=f"up_{i}_{log_id}"):
                        submit_feedback(log_id, "positive")
                        st.session_state["messages"][i]["feedback"] = "positive"
                        st.rerun()
                with col2:
                    if st.button("👎", key=f"down_{i}_{log_id}"):
                        submit_feedback(log_id, "negative")
                        st.session_state["messages"][i]["feedback"] = "negative"
                        st.rerun()

# ── Chat input ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your company documents..."):
    st.session_state.setdefault("messages", [])
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.markdown(
        f'<div class="user-bubble">{prompt}</div>',
        unsafe_allow_html=True,
    )

    # Stream response token by token
    # st.write_stream renders tokens as they arrive — feels like ChatGPT
    with st.container():
        st.markdown('<div class="assistant-bubble">', unsafe_allow_html=True)
        streamed_text = st.write_stream(call_chat_api_streaming(prompt))
        st.markdown('</div>', unsafe_allow_html=True)

    # After streaming completes, get log_id from session state
    stream_result = st.session_state.pop("_last_stream_result", {})
    log_id = stream_result.get("log_id")
    full_answer = stream_result.get("answer", streamed_text or "")

    # Store in message history with log_id for feedback
    st.session_state["messages"].append({
        "role":     "assistant",
        "content":  full_answer,
        "log_id":   log_id,
        "feedback": None,
    })

    st.rerun()