import streamlit as st
import requests
import pyttsx3

st.set_page_config(page_title="🧠 Mental Health Chatbot")
st.title("🧠 Mental Health Support Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_input" not in st.session_state:
    st.session_state.last_input = ""

st.sidebar.header("🎛️ Settings")
voice_out = st.sidebar.checkbox("🔊 Enable Bot Voice")
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.messages = []
    st.session_state.last_input = ""
    st.rerun()

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Display previous messages
for msg in st.session_state.messages:
    role = "🧑 You" if msg["role"] == "You" else "🤖 Bot"
    st.markdown(f"**{role}:** {msg['text']}")

# User input box
user_input = st.text_input("How are you feeling today?")

# Avoid repeat responses
if user_input and st.session_state.last_input != user_input:
    st.session_state.messages.append({"role": "You", "text": user_input})
    st.session_state.last_input = user_input

    history = [m["text"] for m in st.session_state.messages]

    try:
        res = requests.post("http://127.0.0.1:8000/chat/", json={"history": history})
        reply = res.json().get("answer", "I'm sorry, I didn't understand that.")

        # Filter out irrelevant or unexpected answers
        if reply.lower().strip().startswith("it seems you're feeling happy"):
            reply = "😊 That's wonderful to hear. If you'd like to share more, I'm here for you."

        st.session_state.messages.append({"role": "Bot", "text": reply})
        if voice_out:
            speak(reply)
        st.rerun()
    except Exception as e:
        st.error(f"Error communicating with backend: {e}")


