import os
from smolagents import tool
from huggingface_hub import InferenceClient

@tool
def find_emotion(text: str) -> str:
    """
    Analyzes the emotional content of the given text and classifies it.

    Args:
        text (str): The customer's input message.

    Returns:
        str: The detected emotion ("positive", "negative", or "neutral").
    """
        
    emotion_prompt = f"""
    Analyze the emotional content of the following text and classify it as either "positive", "negative", or "neutral".

    Text: "{text}"

    Return only the emotion classification
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
                    "content": emotion_prompt
                }
            ],
        )
        emotion = completion.choices[0].message.content.lower()
        print(f"find_emotion : emotion predicted -{emotion}")
        return emotion
    
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return "neutral" 