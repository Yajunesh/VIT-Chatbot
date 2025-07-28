import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"  # FastAPI backend URL

st.title("VIT Vellore Chatbot")
st.write("Ask questions about VIT Vellore based on official website content.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Your question:", key="input")

if st.button("Ask") and question.strip():
    with st.spinner("Thinking..."):
        try:
            response = requests.post(API_URL, json={"question": question})
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "Sorry, no answer.")
            sources = data.get("sources", [])

            st.session_state.chat_history.append({"user": question, "bot": answer, "sources": sources})

        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        if chat["sources"]:
            st.markdown("**Sources:**")
            for i, src in enumerate(chat["sources"], 1):
                st.markdown(f"{i}. [{src['title']}]({src['url']})")
        st.markdown("---")
