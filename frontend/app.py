import streamlit as st
import requests
from streamlit_chat import message

st.set_page_config(page_title="Dev Docs Copilot", layout="wide")
st.title("CodeSage : Your magical assistant")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of docs to consider (top-k):", 1, 10, 3)
    backend_url = st.text_input("Backend URL:", value="http://localhost:8000")


if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question:")

if st.button("Get Answer") and query.strip() != "":
    with st.spinner("Generating answer..."):
        try:
            res = requests.post(
                f"{backend_url}/ask",
                json={"query": query, "top_k": top_k}
            )
            answer = res.json().get("answer", "No answer returned")
            st.session_state.history.append((query, answer))
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

for q, a in reversed(st.session_state.history):
    message(q, is_user=True)
    message(a, is_user=False)
