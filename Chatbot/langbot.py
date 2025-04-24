import os
import sys
from typing import List, Dict, Any, Optional, TypedDict
from pathlib import Path
import base64
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Install necessary packages if not already installed
try:
    from openai import OpenAI
    from PIL import Image
    import matplotlib.pyplot as plt
    from langgraph.graph import StateGraph, END
except ImportError:
    print("Installing required packages...")
    os.system("pip install pillow matplotlib openai langgraph pydantic python-dotenv")
    from openai import OpenAI
    from PIL import Image
    import matplotlib.pyplot as plt
    from langgraph.graph import StateGraph, END

# Set OpenRouter API key
OPENROUTER_API_KEY = "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"

# Create OpenAI client configured for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Define the state as a TypedDict for LangGraph compatibility
class ChatState(TypedDict):
    """Represents the state of our chatbot application."""
    messages: List[Dict[str, Any]]
    image_path: Optional[str]
    next: Optional[str]

def find_file(filename):
    """Find a file in various locations relative to the script."""
    # Get current script directory
    script_dir = Path(__file__).parent.absolute()
    
    # Try various paths
    paths_to_try = [
        filename,                        # As provided
        f"./{filename}",                 # Relative to current directory
        f"{script_dir}/{filename}",      # Relative to script directory
        str(Path.cwd() / filename)       # Explicit current working directory
    ]
    
    # Check each path
    for path in paths_to_try:
        path_obj = Path(path)
        if path_obj.exists():
            return str(path_obj.absolute())
    
    return None

def encode_image_to_base64(image_path):
    """Convert an image to base64 encoding."""
    if not image_path:
        return None
    
    # Try to find the image file
    resolved_path = find_file(image_path)
    
    if resolved_path:
        try:
            with open(resolved_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
                return encoded
        except Exception:
            return None
    else:
        return None

# Call the Llama-4-Maverick model via OpenRouter
def call_llm(messages, temperature=0.7, max_tokens=300):
    """Call the Llama-4-Maverick model using OpenRouter API directly."""
    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick:free",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers={
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "LangGraph Recommendation Agent",
            }
        )
        return completion.choices[0].message.content
    except Exception as e:
        return "I'm sorry, I encountered an error processing your request."

# Define our agent nodes for proper state handling with LangGraph
def initial_processing(state: ChatState) -> ChatState:
    """Initial processing of the user message, including image detection."""
    # Create a new state object
    new_state = state.copy()
    
    # Determine the next step based on whether there's an image
    if state["image_path"]:
        new_state["next"] = "process_with_image"
    else:
        new_state["next"] = "generate_recommendation"
    
    return new_state

def process_with_image(state: ChatState) -> ChatState:
    """Process the request with an image."""
    # Create a new state object
    new_state = state.copy()
    messages = state["messages"].copy()
    
    # Get the latest human message
    human_messages = [i for i, msg in enumerate(messages) if msg["role"] == "human"]
    
    if human_messages and state["image_path"]:
        idx = human_messages[-1]
        
        # Encode the image
        base64_image = encode_image_to_base64(state["image_path"])
        
        if base64_image:
            # Get the text content
            text_content = messages[idx]["content"]
            
            # Update the message to include the image in OpenAI's format
            messages[idx] = {
                "role": "user",  # OpenAI format uses "user" instead of "human"
                "content": [
                    {"type": "text", "text": text_content},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
    
    new_state["messages"] = messages
    new_state["next"] = "generate_recommendation"
    return new_state

def generate_recommendation(state: ChatState) -> ChatState:
    """Generate recommendations based on user input."""
    # Create a new state object
    new_state = state.copy()
    messages = state["messages"].copy()
    
    # Create a system message for the recommendation assistant
    system_content = """
    You are a helpful AI assistant. When responding to users:

    1. Be conversational, friendly, and natural in your responses
    2. Provide thoughtful answers to the user's questions
    3. If the user shares an image, first describe what you see in the image, then answer any questions about it
    4. Always acknowledge and respond to image content when present
    5. End with a natural follow-up question that continues the conversation

    Maintain a helpful, knowledgeable tone while being concise and direct.
    """
    
    # Convert state messages to OpenAI format
    formatted_messages = [{"role": "system", "content": system_content}]
    
    # Add conversation history - properly handling the conversion between formats
    for msg in messages:
        if msg["role"] == "human":
            if isinstance(msg.get("content"), list):  # If it's already in multimodal format
                formatted_messages.append({"role": "user", "content": msg["content"]})
            else:  # If it's just text
                formatted_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "ai":
            formatted_messages.append({"role": "assistant", "content": msg["content"]})
        elif msg["role"] == "user":  # Handle already converted messages
            formatted_messages.append(msg)
        elif msg["role"] == "assistant":  # Handle already converted messages
            formatted_messages.append(msg)
    
    # Get the response from the LLM
    response = call_llm(formatted_messages, temperature=0.7, max_tokens=500)
    
    # Add the AI response to the messages - convert back to internal format
    messages.append({"role": "ai", "content": response})
    
    new_state["messages"] = messages
    new_state["next"] = END
    new_state["image_path"] = None  # Clear the image path for the next interaction
    return new_state

# Build the agent graph
def build_agent_graph():
    """Build and return the LangGraph agent."""
    # Create the graph with a dictionary-based state
    builder = StateGraph(ChatState)
    
    # Add nodes
    builder.add_node("initial_processing", initial_processing)
    builder.add_node("process_with_image", process_with_image)
    builder.add_node("generate_recommendation", generate_recommendation)
    
    # Add conditional edges based on the 'next' field
    builder.add_conditional_edges(
        "initial_processing",
        lambda state: state["next"]
    )
    
    builder.add_conditional_edges(
        "process_with_image",
        lambda state: state["next"]
    )
    
    # Set the entry point
    builder.set_entry_point("initial_processing")
    
    # Compile the graph
    return builder.compile()

# This function is kept but modified to not display the image
def display_image(image_path: str):
    """Validate the image file exists but don't display it."""
    if not image_path:
        return False
    
    path_obj = Path(image_path)
    return path_obj.exists()

# Main program
def main():
    """Main function to run the chatbot."""
    print("Agentic Chatbot with Llama-4-Maverick")
    print("Type 'exit' to quit")
    print("To share an image, type 'image:image_address' (with no space after colon)\n")
    
    # Build the agent graph
    agent = build_agent_graph()
    
    # Initialize messages history
    messages_history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        
        # Initialize the state for this interaction
        image_path = None
        
        # Check for image in user input - allow for both "image:" and "image: " formats
        if "image:" in user_input:
            # Parse image path
            parts = user_input.split("image:")
            user_input = parts[0].strip()
            image_path = parts[1].strip()
            
            # Find the image file
            resolved_path = find_file(image_path)
            if resolved_path:
                image_path = resolved_path
                print(f"Image received: {Path(image_path).name}")
                # We validate the image exists but don't display it
                if not display_image(image_path):
                    print(f"Warning: Image file exists but may not be a valid image format")
            else:
                print(f"Image not found: {image_path}")
                image_path = None
        
        # If user input is empty after extracting image, use a generic prompt
        if not user_input and image_path:
            user_input = "What can you tell me about this image?"
        
        # Create current message history by copying and adding new message
        current_messages = messages_history.copy()
        current_messages.append({"role": "human", "content": user_input})
        
        # Create state as a dictionary
        current_state = {
            "messages": current_messages,
            "image_path": image_path,
            "next": None
        }
        
        try:
            print("Chatbot is thinking...", end="", flush=True)
            # Run the agent on the current state
            new_state = agent.invoke(current_state)
            
            # Update the global message history - ensure we're storing the properly formatted messages
            messages_history = new_state["messages"]
            
            print("\r" + " " * 25 + "\r", end="")  # Clear the "thinking" message
            
            # Print the response
            last_ai_message = next((msg for msg in reversed(messages_history) if msg["role"] == "ai"), None)
            if last_ai_message:
                print(f"\nChatbot: {last_ai_message['content']}\n")
            else:
                print("Chatbot: I'm sorry, I couldn't generate a response at this time.")
        except Exception:
            print("\r" + " " * 25 + "\r", end="")  # Clear the "thinking" message
            print("Chatbot: I'm sorry, I encountered an error processing your request. Please try again.")
            
if __name__ == "__main__":
    main()