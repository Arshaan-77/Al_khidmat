from backend3 import workflow
from database import save_beneficiaries

from langchain_core.messages import HumanMessage

import streamlit as st
import uuid


# -----------------------------
# Session State Initialization
# -----------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "final_data" not in st.session_state:
    st.session_state.final_data = None


# -----------------------------
# LangGraph Config
# -----------------------------

CONFIG = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# -----------------------------
# Page Title
# -----------------------------

st.title("AI-Khidmat Registration Assistant")
st.subheader("""AI-Khidmat is a platform designed to help people in need by connecting beneficiaries, volunteers, and donors.""")
st.subheader("Mandatory fields before completion:") 
st.text("- name \n- phone \n- need")
st.subheader("Optional Fields:")
st.text("- Location \n-Family Members")
# -----------------------------
# Display Chat History
# -----------------------------

for message in st.session_state.message_history:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# -----------------------------
# Chat Input
# -----------------------------

user_input = st.chat_input(
    "Enter your details..."
)


# -----------------------------
# Handle User Input
# -----------------------------

if user_input:

    # Store User Message
    st.session_state.message_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Invoke Workflow
    response = workflow.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config=CONFIG
    )

    ai_message = response["messages"][-1].content

    # Store Assistant Message
    st.session_state.message_history.append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )

    # Store extracted data temporarily
    if "Registration Completed" in ai_message:

        st.session_state.final_data = response["extracted_data"]

    st.rerun()


# -----------------------------
# Save Button Section
# -----------------------------

if st.session_state.final_data is not None:

    st.divider()

    if st.button("Save Information"):

        save_beneficiaries(
            st.session_state.final_data
        )

        st.success(
            "Information saved successfully!"
        )

        # Reset for new user
        st.session_state.thread_id = str(uuid.uuid4())

        st.session_state.message_history = []

        st.session_state.final_data = None

        st.rerun()