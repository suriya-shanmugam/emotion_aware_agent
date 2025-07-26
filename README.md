# RAG-Enhanced Emotion-Aware Support Agent

A sophisticated customer support assistant that combines emotion analysis with Retrieval-Augmented Generation (RAG) capabilities. This project demonstrates how to build an intelligent conversational AI agent that can:

- Detect customer emotions from chat messages
- Log and monitor emotional trends
- Trigger human agent intervention on repeated negative emotions
- Generate context-aware responses using uploaded PDF documents
- Provide a modern, interactive web UI using Streamlit
- Maintain a persistent vector store for document knowledge

## Features

- **RAG Integration:** Retrieval-Augmented Generation using FAISS vector store with HuggingFace embeddings
- **PDF Document Processing:** Automatic processing and indexing of uploaded PDF files
- **Modular Tooling:** Each tool (emotion detection, logging, response) is implemented as a separate, reusable Python module
- **LLM Integration:** Uses Llama 3 (via HuggingFace Inference API) for emotion analysis and response generation
- **Human Agent Trigger:** Automatically flags for human intervention if three consecutive negative emotions are detected
- **Streamlit UI:** Real-time chat interface with emotion monitoring dashboard and file upload section
- **Vector Store Management:** Persistent document embeddings with automatic updates and state management
- **Environment-based Secrets:** API keys are loaded from environment variables for security

## Directory Structure

```
hello_agent/
├── app.py                # Main Streamlit app with RAG integration
├── vector_store.py       # Vector store manager for document embeddings
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
   ```bash
   export HF_TOKEN=YOUR_HUGGINGFACE_API_TOKEN
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## RAG System Architecture

The application uses a sophisticated RAG (Retrieval-Augmented Generation) system:

### Vector Store Manager (`vector_store.py`)
- **Embedding Model**: Uses `thenlper/gte-small` for document embeddings
- **Vector Database**: FAISS with cosine similarity for efficient retrieval
- **Document Processing**: Automatic chunking with 512-token windows and overlap
- **State Management**: Tracks processed files to avoid reprocessing

### Document Processing Pipeline
1. **Upload**: PDF files are uploaded via Streamlit interface
2. **Processing**: Documents are chunked and embedded using HuggingFace transformers
3. **Indexing**: Chunks are stored in FAISS vector store with metadata
4. **Retrieval**: Relevant documents are retrieved based on semantic similarity
5. **Generation**: LLM generates responses using retrieved context

## Usage

The application features a three-panel layout:

- **Left Panel (Customer Monitoring):** Shows the last 10 detected emotions and triggers a human agent alert if needed
- **Middle Panel (Chat Interface):** Interactive chat with the assistant that analyzes emotion, logs it, and generates context-aware responses using uploaded documents
- **Right Panel (File Upload):** Upload PDF files that are automatically processed and indexed for RAG-based responses

### Workflow
1. **Upload Documents**: Upload PDF files in the right panel - they're automatically processed and indexed
2. **Ask Questions**: Chat with the assistant in the middle panel
3. **Get Informed Responses**: The assistant uses uploaded documents to provide accurate, context-aware answers
4. **Monitor Emotions**: Track customer emotions in the left panel

## Tool Modularity

- `tools/find_emotion.py`: Uses Llama LLM to classify emotion from text
- `tools/log_emotion.py`: Logs emotion to session state and manages human agent trigger logic
- `tools/respond_customer.py`: Uses Llama LLM to generate context-aware responses using retrieved documents from the vector store

## Key Components

### Vector Store Manager
- **Automatic Updates**: Vector store updates when new PDFs are uploaded
- **Duplicate Prevention**: Tracks processed files to avoid reprocessing
- **Metadata Tracking**: Maintains source file information for each document chunk
- **Efficient Retrieval**: Uses cosine similarity for semantic search

### Response Generation
- **Context-Aware**: Responses are generated using relevant document content
- **Semantic Search**: Retrieves most relevant document chunks for each query
- **Fallback Handling**: Gracefully handles cases when no relevant documents are found

## Customization

- **LLM Models**: You can swap out the LLM model in `app.py` or tool files as needed
- **Embedding Models**: Change the embedding model in `vector_store.py` by modifying `EMBEDDING_MODEL_NAME`
- **Chunking Parameters**: Adjust chunk size and overlap in `vector_store.py`
- **Tools**: Add more tools to the `tools/` directory and register them with the agent for new capabilities

## Technical Details

### Dependencies
- **LangChain**: Document processing and vector store management
- **FAISS**: Efficient similarity search and indexing
- **HuggingFace Transformers**: Embedding models and tokenization
- **PyTorch**: Deep learning framework for embeddings
- **Streamlit**: Web interface and file upload handling

### Performance Considerations
- **GPU Support**: Automatically uses CUDA if available for faster embedding generation
- **Memory Management**: Efficient document chunking and processing
- **Session Persistence**: Vector store maintains state across Streamlit sessions

## Security

- **Never commit API keys** to source control. Always use environment variables for secrets
- **File Handling**: Uploaded files are stored in temporary locations and cleaned up properly


