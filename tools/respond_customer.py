import streamlit as st
import os
from smolagents import tool
from huggingface_hub import InferenceClient
from vector_store import vector_store_manager

@tool
def respond_customer(query: str) -> str:
    """
    Generates a helpful customer response based on the query and relevant documents from vector store.

    Args:
        query (str): The customer's question or complaint.
        

    Returns:
        str: The assistant's response by relevant documents.
    """
    # Fetch relevant documents from vector store
    related_docs = vector_store_manager.get_relevant_docs(query, k=3)
    
    # Prepare context from documents
    if related_docs:
        retrieved_docs_text = [doc.page_content for doc in related_docs]  # We only need the text of the documents
        context = "\nExtracted documents:\n"
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])
    else:
        context = "No relevant documents found in uploaded PDFs."
    
    query_context = f"""
        
        
        Customer Query: "{query}"
        context : "{context}"

        """
    
    try:
        client = InferenceClient(
            provider="groq",
            api_key=os.environ["HF_TOKEN"],
        )

        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                 {
                    "role": "system",
                    "content": """Guidelines:
                                    - Use information from the provided context to give accurate and helpful responses
                                    - If the answer cannot be deduced from the context, acknowledge that and offer to help with other queries
                                    - Don't mention that context is provided to you

                                """,
                },
                {
                    "role": "user",
                    "content": query_context
                }
            ],
        )
        reply = completion.choices[0].message.content.lower()
        print(f"respond_customer : reply to customer -{reply}")

        
    except Exception as e:
        print(f"Error in response generation: {e}")
        reply = f"I understand your query about {query}. Let me help you with that."

    # Update chat history within the tool
    chat_history = st.session_state['chat_history']
    chat_history.append({"user": query, "assistant": reply})
    st.session_state['chat_history'] = chat_history

    return reply 