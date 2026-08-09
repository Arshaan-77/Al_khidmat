from backend5 import chatbot

from database import (save_beneficiaries2, save_face_embedding, check_duplicate_in_database)


from face_embeddings import validate_face

from langchain_core.messages import HumanMessage

import streamlit as st
import uuid
import os


# ==================================
# Session State
# ==================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "final_data" not in st.session_state:
    st.session_state.final_data = None


# ==================================
# LangGraph Config
# ==================================

CONFIG = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# ==================================
# Title
# ==================================

st.title("AI-Beneficiary Registration Assistant")

st.subheader(
    """
AI-Khidmat is a platform designed to help people in need
by connecting beneficiaries, volunteers, and donors.
"""
)


# ==================================
# Chat History
# ==================================

for message in st.session_state.message_history:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ==================================
# User Input
# ==================================

user_input = st.chat_input(
    "Enter your details..."
)

if user_input:

    st.session_state.message_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    response = chatbot.invoke(
        {
            "messages": [
                HumanMessage(
                    content=user_input
                )
            ]
        },
        config=CONFIG
    )

    ai_message = response["messages"][-1].content

    st.session_state.message_history.append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )

    if "Registration Completed" in ai_message:

        st.session_state.final_data = {
            "beneficiary":
                response["extracted_dict"],

            "medical":
                response.get(
                    "medical",
                    {}
                ),

            "education":
                response.get(
                    "education",
                    {}
                ),

            "financial":
                response.get(
                    "financial",
                    {}
                )
        }

    st.rerun()


# ==================================
# Face Verification Section
# ==================================

if st.session_state.final_data is not None:

    st.divider()

    st.subheader(
        "Face Verification"
    )

    uploaded_file = st.file_uploader(
        "Upload Beneficiary Photo",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="beneficiary_photo"
    )

    camera_photo = st.camera_input(
        "Or Click Photo",
        key="beneficiary_camera"
    )

    image_file = None

    if uploaded_file is not None:
        image_file = uploaded_file

    elif camera_photo is not None:
        image_file = camera_photo

    if st.button(
        "Verify & Save Information"
    ):

        if image_file is None:

            st.error(
                "Please upload or capture a photo."
            )

        else:

            os.makedirs(
                "uploads/beneficiaries",
                exist_ok=True
            )

            image_path = os.path.join(
                "uploads/beneficiaries",
                f"{uuid.uuid4()}.jpg"
            )

            with open(
                image_path,
                "wb"
            ) as f:

                f.write(
                    image_file.getbuffer()
                )

            # ==========================
            # Face Validation
            # ==========================

            is_valid, message = validate_face(
                image_path
            )

            if not is_valid:

                st.error(message)

                try:
                    os.remove(image_path)
                except:
                    pass

            else:

                # ==========================
                # Duplicate Check
                # ==========================

                duplicate_result = (
                    check_duplicate_in_database(
                        image_path
                    )
                )

                if duplicate_result.get(
                    "duplicate"
                ):

                    st.error(
                        f"""
Duplicate Face Detected

Existing Beneficiary ID:
{duplicate_result['beneficiary_id']}

Similarity:
{duplicate_result['similarity']}
"""
                    )

                    try:
                        os.remove(image_path)
                    except:
                        pass

                else:

                    # ==========================
                    # Save Beneficiary
                    # ==========================

                    beneficiary_id = (
                        save_beneficiaries2(
                            st.session_state.final_data
                        )
                    )

                    if beneficiary_id is None:

                        st.error(
                            "Unable to save beneficiary."
                        )

                    else:

                        # ==========================
                        # Save Embedding
                        # ==========================

                        result = (
                            save_face_embedding(
                                beneficiary_id,
                                image_path
                            )
                        )

                        if not result.get(
                            "success"
                        ):

                            st.error(
                                result.get(
                                    "error",
                                    "Embedding save failed."
                                )
                            )

                        else:

                            st.success(
                                "Registration completed successfully."
                            )

                            # ==========================
                            # Reset Session
                            # ==========================

                            st.session_state.thread_id = (
                                str(
                                    uuid.uuid4()
                                )
                            )

                            st.session_state.message_history = []

                            st.session_state.final_data = None

                            st.rerun()