from smolagents import OpenAIServerModel, LiteLLMModel, CodeAgent, tool

from collections import deque

@tool
def emotion_monitor(status:str) -> str :
    """
    Monitors and logs the emotional state based on the given status.

    Args:
        status (str): The emotional status to monitor (e.g., "positive", "negative", "neutral").

    Returns:
        None
    """ 
    
    if status == 'positive' :
        return "Keep the conversation going"
    elif status == 'neutral' :
        return "Carefull with conversation"
    else : 
        return "Find the human agent immediately!!!"


model = LiteLLMModel(
        model_id="ollama_chat/qwen2:7b",  # Or try other Ollama-supported models
        api_base="http://127.0.0.1:11434",  # Default Ollama local server
        num_ctx=8192,)

#model = OpenAIServerModel(
#    model_id="gemini-2.0-flash",
    # Google Gemini OpenAI-compatible API base URL
#    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
#    api_key="",
#)


agent = CodeAgent(tools=[emotion_monitor], model=model, add_base_tools=True)
agent.run("<<< This sounds great!!! >>>' Catagorize  on emotion of the sentence in <<< >>> by yourself  classify it into positive, negative and neutral. Pass that as a status argument to emotion_monitor to suggest next step")

