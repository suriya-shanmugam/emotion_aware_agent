# Emotion-Aware Support Agent

A modular, emotion-aware customer support assistant built with Streamlit and LLM tools. This project demonstrates how to build a conversational AI agent that can:

- Detect customer emotions from chat messages
- Log and monitor emotional trends
- Trigger human agent intervention on repeated negative emotions
- Generate empathetic, context-aware responses
- Provide a modern, interactive web UI using Streamlit

## Features

- **Modular Tooling:** Each tool (emotion detection, logging, response) is implemented as a separate, reusable Python module.
- **LLM Integration:** Uses Llama 3 (via HuggingFace Inference API) for emotion analysis and response generation.
- **Human Agent Trigger:** Automatically flags for human intervention if three consecutive negative emotions are detected.
- **Streamlit UI:** Real-time chat interface with emotion monitoring dashboard.
- **Environment-based Secrets:** API keys are loaded from environment variables for security.

## Directory Structure

```
agents/
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── tools/                # Modular tool implementations
│   ├── __init__.py
│   ├── find_emotion.py
│   ├── log_emotion.py
│   └── respond_customer.py
└── README.md             # Project documentation
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd agents
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables:**
   - For Gemini (Google):
     ```bash
     export gemini_key=YOUR_GEMINI_API_KEY
     ```
   - For Llama (HuggingFace):
     ```bash
     export HF_TOKEN=YOUR_HUGGINGFACE_API_TOKEN
     ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Usage

- **Left Panel:** Shows the last 10 detected emotions and triggers a human agent alert if needed.
- **Right Panel:** Chat interface for user and assistant. The assistant analyzes emotion, logs it, and generates a response using LLM tools.

## Tool Modularity

- `tools/find_emotion.py`: Uses Llama LLM to classify emotion from text.
- `tools/log_emotion.py`: Logs emotion to session state and manages human agent trigger logic.
- `tools/respond_customer.py`: Uses Llama LLM to generate a context-aware, empathetic response and updates chat history.

## Customization

- You can swap out the LLM model in `app.py` or tool files as needed.
- Add more tools to the `tools/` directory and register them with the agent for new capabilities.

## Security

- **Never commit API keys** to source control. Always use environment variables for secrets.


