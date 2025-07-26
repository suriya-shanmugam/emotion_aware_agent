import streamlit as st
import os
import tempfile
from smolagents import OpenAIServerModel, LiteLLMModel, CodeAgent, tool, ChatMessage

from huggingface_hub import InferenceClient

# Import individual tools from separate files
from tools.find_emotion import find_emotion
from tools.log_emotion import log_emotion
from tools.respond_customer import respond_customer

# Import vector store manager
from vector_store import vector_store_manager

#--- LLM Initialization ---
model = LiteLLMModel(
    model_id="ollama_chat/qwen2:latest",  # Use your Ollama model here
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)

# model = OpenAIServerModel(
#      model_id="gemini-2.0-flash",
#      # Google Gemini OpenAI-compatible API base URL
#      api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
#      api_key=os.getenv("gemini_key"),
# )

# --- State Management ---
if 'previous_emotions' not in st.session_state:
    st.session_state['previous_emotions'] = []
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'trigger_human_agent' not in st.session_state:
    st.session_state['trigger_human_agent'] = False
if 'clear_input' not in st.session_state:
    st.session_state['clear_input'] = False
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []

WINDOW_SIZE = 3

def update_emotion(emotion):
    st.session_state['previous_emotions'].append(emotion)
    last_three = st.session_state['previous_emotions'][-WINDOW_SIZE:]
    trigger = all(e == 'negative' for e in last_three) and len(last_three) == WINDOW_SIZE
    st.session_state['trigger_human_agent'] = trigger

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location and store info in session state"""
    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Store file info in session state
        file_info = {
            'name': uploaded_file.name,
            'size': uploaded_file.size,
            'type': uploaded_file.type,
            'path': tmp_file_path
        }
        st.session_state['uploaded_files'].append(file_info)
        return file_info
    return None

# --- Agent Setup ---
agent = CodeAgent(
    tools=[find_emotion, log_emotion, respond_customer],
    model=model,
    add_base_tools=True,
    max_steps=5
)

# --- Streamlit Layout ---
st.set_page_config(layout="wide")
col1, col2, col3 = st.columns([1, 2, 1])

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
3. Use respond_customer tool to repond to customer query.

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

with col3:
    st.header("📁 File Upload")
    st.write("Upload PDF files for reference")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type=['pdf'], 
        key="pdf_uploader",
        help="Upload a PDF file to store in session memory"
    )
    
    # Handle file upload - automatically save to session
    if uploaded_file is not None:
        # Check if file is already uploaded to avoid duplicates
        file_names = [f['name'] for f in st.session_state['uploaded_files']]
        if uploaded_file.name not in file_names:
            file_info = save_uploaded_file(uploaded_file)
            if file_info:
                st.success(f"✅ File '{file_info['name']}' automatically saved to session!")
                # Update vector store with new file
                vector_store_manager.update_vector_store(st.session_state['uploaded_files'])
                st.rerun()
        else:
            st.info(f"File '{uploaded_file.name}' is already uploaded")
    
    # Display uploaded files
    if st.session_state['uploaded_files']:
        st.subheader("📋 Uploaded Files")
        
        # Show vector store status
        vector_info = vector_store_manager.get_vector_store_info()
        if vector_info['status'] == 'active':
            st.success(f"🔍 Vector Store: {vector_info['total_documents']} documents indexed")
        else:
            st.info("🔍 Vector Store: No documents indexed yet")
        
        for i, file_info in enumerate(st.session_state['uploaded_files']):
            with st.expander(f"📄 {file_info['name']} ({file_info['size']} bytes)"):
                st.write(f"**File Type:** {file_info['type']}")
                st.write(f"**File Path:** {file_info['path']}")
                
                # Option to remove file
                if st.button(f"Remove {file_info['name']}", key=f"remove_{i}"):
                    # Clean up temporary file
                    try:
                        os.unlink(file_info['path'])
                    except:
                        pass
                    # Remove from session state
                    st.session_state['uploaded_files'].pop(i)
                    # Rebuild vector store with remaining files
                    vector_store_manager.clear_vector_store()
                    if st.session_state['uploaded_files']:
                        vector_store_manager.update_vector_store(st.session_state['uploaded_files'])
                    st.rerun()
    else:
        st.info("No files uploaded yet")

