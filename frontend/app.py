"""
frontend/app.py - NexusAI Login Page
Employees login with employee_id + password
"""

import streamlit as st
import requests
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

API_BASE = os.getenv("API_BASE_URL", "http://localhost:5050")

st.set_page_config(
    page_title="NexusAI - Login",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stSidebarNav"] { display: none; }
    .logo-text {
        font-size: 2.5rem;
        font-weight: 700;
        color: #6C63FF;
        text-align: center;
    }
    .logo-sub {
        font-size: 1rem;
        color: #9B9DB5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hint-box {
        background: #13151F;
        border: 1px solid #2D2F3E;
        border-left: 3px solid #6C63FF;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: #6B6D85;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    .stButton > button {
        width: 100%;
        background-color: #6C63FF;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        font-size: 1rem;
    }
    .stButton > button:hover {
        background-color: #5A52D5;
        color: white;
    }
    .error-msg {
        background-color: #2D1A1A;
        border: 1px solid #FF4B4B;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        color: #FF4B4B;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

if st.session_state.get("logged_in"):
    st.switch_page("pages/chat.py")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="logo-text">🧠 NexusAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="logo-sub">Your intelligent company knowledge assistant</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Login hint
    st.markdown("""
    <div class="hint-box">
        Login with your <strong style="color:#9D97F5;">Employee ID</strong>
        and default password.<br>
        Default password: <strong style="color:#9D97F5;">EMP + last 4 digits of your ID + @Nexus</strong><br>
        Example: ID <code>FINEMP1012</code> → password <code>EMP1012@Nexus</code>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=False):
        employee_id = st.text_input(
            "Employee ID",
            placeholder="e.g. FINEMP1012",
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="EMP1012@Nexus",
        )
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Sign In", use_container_width=True)

    if submitted:
        employee_id = employee_id.strip().upper()
        password = password.strip()

        if not employee_id or not password:
            st.markdown(
                '<div class="error-msg">Please enter your Employee ID and password.</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Signing in..."):
                try:
                    response = requests.post(
                        f"{API_BASE}/auth/login",
                        json={"employee_id": employee_id, "password": password},
                        headers={"Content-Type": "application/json"},
                        timeout=10,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state["logged_in"]   = True
                        st.session_state["token"]       = data["access_token"]
                        st.session_state["user_id"]     = data["employee_id"]
                        st.session_state["user_name"]   = data["name"]
                        st.session_state["user_email"]  = data.get("email", "")
                        st.session_state["user_role"]   = data["role"]
                        st.session_state["department"]  = data["department"]
                        st.session_state["messages"]    = []
                        st.switch_page("pages/chat.py")
                    elif response.status_code == 401:
                        st.markdown(
                            '<div class="error-msg">Invalid Employee ID or password.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        try:
                            detail = response.json().get("detail", response.text)
                        except Exception:
                            detail = response.text
                        st.markdown(
                            f'<div class="error-msg">Error {response.status_code}: {detail}</div>',
                            unsafe_allow_html=True,
                        )
                except requests.exceptions.ConnectionError:
                    st.markdown(
                        '<div class="error-msg">Cannot connect to server. Make sure backend is running on port 5050.</div>',
                        unsafe_allow_html=True,
                    )
                except requests.exceptions.Timeout:
                    st.markdown(
                        '<div class="error-msg">Request timed out. Please try again.</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center; color:#4A4D6A; font-size:0.8rem;">NexusAI v1.0 - Internal Use Only</div>',
        unsafe_allow_html=True,
    )