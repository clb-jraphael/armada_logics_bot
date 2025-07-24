# app.py

import streamlit as st
from chatbot import answer_query

st.set_page_config(page_title="Gemma Chatbot", layout="centered")
st.title("🤖 Armadalogics Chatbot")

st.write("Ask anything about your documents.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your question here...")

if user_input:
    with st.spinner("Thinking..."):
        answer, sources = answer_query(user_input)
        st.session_state.chat_history.append(
            {"q": user_input, "a": answer, "sources": sources}
        )

for entry in st.session_state.chat_history:
    st.chat_message("user").write(entry["q"])
    assistant_box = st.chat_message("assistant")
    assistant_box.write(entry["a"])
    if entry["sources"]:
        with assistant_box.expander("Sources"):
            for s in entry["sources"]:
                line = f"**{s['source']}**"
                if s["category"]:
                    line += f" (Category: {s['category']})"
                if s["page"] is not None:
                    line += f", Page {s['page']}"
                st.write(line)
                st.caption(s["snippet"])
