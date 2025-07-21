import streamlit as st
import os
from smolagents import tool
from huggingface_hub import InferenceClient

@tool
def respond_customer(query: str, emotion: str) -> str:
    """
    Generates a helpful customer response based on the query and detected emotion.

    Args:
        query (str): The customer's question or complaint.
        emotion (str): The detected emotion of the customer.

    Returns:
        str: The assistant's response tailored to the emotional tone.
    """
    response_prompt = f"""
You are a helpful customer support assistant. Generate an appropriate response to the customer's query.

Customer Query: "{query}"
Detected Emotion: {emotion}

Guidelines:
- If emotion is "negative": Show empathy, acknowledge their frustration, and provide helpful assistance
- If emotion is "positive": Be enthusiastic and supportive while addressing their query
- If emotion is "neutral": Provide clear, professional assistance

Generate a natural, helpful response that addresses their query while being appropriate for their emotional state.
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
                    "role": "user",
                    "content": response_prompt
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