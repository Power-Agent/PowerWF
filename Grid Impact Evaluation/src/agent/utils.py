from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_conversation_history(messages: List[AnyMessage]) -> str:
    """
    Get a formatted conversation history from the messages.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        Formatted string representation of the conversation
    """
    if len(messages) == 1:
        return messages[-1].content
    else:
        conversation = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                conversation += f"Assistant: {message.content}\n"
        return conversation
