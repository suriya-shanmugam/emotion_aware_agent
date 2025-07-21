import streamlit as st
import os
from smolagents import OpenAIServerModel, LiteLLMModel, CodeAgent, tool, ChatMessage

from huggingface_hub import InferenceClient

# Import individual tools from separate files
from tools.find_emotion import find_emotion
from tools.log_emotion import log_emotion
from tools.respond_customer import respond_customer

# --- LLM Initialization ---
# model = LiteLLMModel(
#     model_id="ollama_chat/qwen2:7b",  # Use your Ollama model here
#     api_base="http://127.0.0.1:11434",
#     num_ctx=8192,
# )

model = OpenAIServerModel(
     model_id="gemini-2.0-flash",
     # Google Gemini OpenAI-compatible API base URL
     api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
     api_key=os.getenv("gemini_key"),
)

# --- State Management ---
if 'previous_emotions' not in st.session_state:
    st.session_state['previous_emotions'] = []
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'trigger_human_agent' not in st.session_state:
    st.session_state['trigger_human_agent'] = False
if 'clear_input' not in st.session_state:
    st.session_state['clear_input'] = False

WINDOW_SIZE = 3

def update_emotion(emotion):
    st.session_state['previous_emotions'].append(emotion)
    last_three = st.session_state['previous_emotions'][-WINDOW_SIZE:]
    trigger = all(e == 'negative' for e in last_three) and len(last_three) == WINDOW_SIZE
    st.session_state['trigger_human_agent'] = trigger


# --- Agent Setup ---
agent = CodeAgent(
    tools=[find_emotion, log_emotion, respond_customer],
    model=model,
    add_base_tools=True,
    max_steps=5
)

# --- Streamlit Layout ---
st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Customer Monitoring")
    st.write("Last 10 Emotions:")
    st.write(st.session_state['previous_emotions'][-10:])
    if st.session_state['trigger_human_agent']:
        st.error("⚠️ Triggered Human Agent! (Last 3 emotions were negative)")
    else:
        st.success("✅ Monitoring...")

with col2:
    st.header("Emotion-Aware Support Assistant")
    chat_placeholder = st.empty()
    chat_history = st.session_state['chat_history']
    for entry in chat_history:
        st.markdown(f"**You:** {entry['user']}")
        st.markdown(f"**Assistant:** {entry['assistant']}")

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message and press Enter", key="user_input")
        submitted = st.form_submit_button("Send")
        if submitted and user_input:
            task = f"""
You're a Support Assistant. You respond to the customer resolving the query.

Steps:
1. Use find_emotion tool to analyze the emotion in the customer's message.
2. Use log_emotion tool to log the detected emotion.
3. Use respond_customer tool to generate a response.

Customer Message:
\"\"\"{user_input}\"\"\"
"""
            try:
                response = agent.run(task)
                agent_response = str(response)
            except Exception as e:
                print("Agent Error:", e)
                agent_response = "❌ Issue with the Agent."

            st.session_state['clear_input'] = True
            st.rerun()

