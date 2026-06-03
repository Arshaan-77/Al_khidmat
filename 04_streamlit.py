# streamlit_app.py

import uuid
import json
import streamlit as st

from langchain_core.messages import HumanMessage
from backend4 import chatbot
from database2 import create_tables, save_registration

# =========================================================
# ONE-TIME SETUP
# =========================================================

create_tables()

# =========================================================
# SESSION STATE
# =========================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "final_data" not in st.session_state:
    st.session_state.final_data = None

if "saved_to_db" not in st.session_state:
    st.session_state.saved_to_db = False        # tracks whether saved already

if "beneficiary_id" not in st.session_state:
    st.session_state.beneficiary_id = None

# =========================================================
# LANGGRAPH CONFIG
# =========================================================

CONFIG = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI-Khidmat Registration Assistant",
    page_icon="🤝",
    layout="centered"
)

# =========================================================
# HEADER
# =========================================================

st.title("🤝 AI-Khidmat Registration Assistant")

st.subheader(
    "AI-Khidmat connects beneficiaries, volunteers, and donors "
    "to help people in need."
)

# =========================================================
# REQUIRED FIELDS INFO
# =========================================================

with st.expander("📋 Required Information"):
    st.markdown("""
### Medical
- Disease · Hospital · Urgency · Treatment Cost

### Education
- Student Class · Institute · Academic Status

### Financial
- Monthly Income · Employment Status · Earning Members

### Basic Information
- Name · Phone · Location · Family Members
""")

# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# =========================================================
# CHAT INPUT  (disabled once registration is complete)
# =========================================================

chat_disabled = st.session_state.final_data is not None

user_input = st.chat_input(
    "Type your message...",
    disabled=chat_disabled,
)

# =========================================================
# HANDLE USER INPUT
# =========================================================

if user_input:

    st.session_state.message_history.append({
        "role": "user",
        "content": user_input,
    })

    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=CONFIG,
    )

    ai_message = response["messages"][-1].content

    st.session_state.message_history.append({
        "role": "assistant",
        "content": ai_message,
    })

    registration_status = response.get("registration_status", "")
    extracted_data      = response.get("extracted_dict", {})

    if registration_status == "complete":
        st.session_state.final_data = extracted_data
        st.session_state.message_history.append({
            "role": "assistant",
            "content": "✅ Registration completed! Please review your information below and click **Save Registration** when ready.",
        })

    st.rerun()

# =========================================================
# FINAL DATA SECTION
# =========================================================

if st.session_state.final_data is not None:

    st.divider()
    st.success("✅ Registration Completed")
    st.subheader("📄 Review Your Information")

    st.json(st.session_state.final_data)

    st.divider()

    # ── Already saved confirmation ────────────────────────────────────────
    if st.session_state.saved_to_db:
        st.success(
            f"💾 Saved to database — Beneficiary ID: **{st.session_state.beneficiary_id}**"
        )

    # ── Save button (hidden after saving) ─────────────────────────────────
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            if st.button("💾 Save Registration", type="primary", use_container_width=True):
                try:
                    bid = save_registration(st.session_state.final_data)
                    st.session_state.beneficiary_id = bid
                    st.session_state.saved_to_db    = True
                    st.rerun()
                except Exception as e:
                    st.error(f"⚠️ Could not save to database: {e}")

        with col2:
            st.caption("Your information will be stored securely in our database.")

    st.divider()

    # ── Download JSON ──────────────────────────────────────────────────────
    json_data = json.dumps(
        st.session_state.final_data,
        indent=4,
        ensure_ascii=False,
    )

    st.download_button(
        label="⬇ Download as JSON",
        data=json_data,
        file_name="beneficiary_data.json",
        mime="application/json",
        use_container_width=True,
    )

    # ── New registration ───────────────────────────────────────────────────
    if st.button("🔄 Start New Registration", use_container_width=True):
        st.session_state.thread_id      = str(uuid.uuid4())
        st.session_state.message_history = []
        st.session_state.final_data     = None
        st.session_state.saved_to_db    = False
        st.session_state.beneficiary_id = None
        st.rerun()