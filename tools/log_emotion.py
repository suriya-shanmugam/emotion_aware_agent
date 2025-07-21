import streamlit as st
from smolagents import tool

def update_emotion(emotion):
    """Helper function to update emotion tracking in session state"""
    st.session_state['previous_emotions'].append(emotion)
    last_three = st.session_state['previous_emotions'][-3:]  # WINDOW_SIZE = 3
    trigger = all(e == 'negative' for e in last_three) and len(last_three) == 3
    st.session_state['trigger_human_agent'] = trigger

@tool
def log_emotion(emotion: str) -> None:
    """
    Logs the detected emotion into session state for tracking.

    Args:
        emotion (str): The emotion to log ("positive", "negative", or "neutral").

    Returns:
        None
    """
    update_emotion(emotion) 