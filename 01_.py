import streamlit as st

st.header("Al_Khidmat Registration")

name = st.text_input("Name", placeholder="Enter your full name")
phone = st.text_input("Phone No.")
st.text_area("Location")
st.text_input("Famil_members")
need = st.text_area("Need")

if st.button("Submit"):
    if not name.strip():
        st.error("Name is required")

    elif not phone.strip():
        st.error("Phone number is required")

    elif not need.strip():
        st.error("Need field is required")

    else:
        st.success("Success")