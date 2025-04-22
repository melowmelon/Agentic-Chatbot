import os
import sys
from typing import List, Optional, Dict, Any, TypedDict, Literal
from pathlib import Path

# Install necessary packages if not already installed
try:
    from langchain_core.messages import HumanMessage, AIMessage
    from PIL import Image
    import matplotlib.pyplot as plt
    from langchain_ollama import OllamaLLM
except ImportError:
    print("Installing required packages...")
    os.system("pip install pillow matplotlib langchain-core langchain-ollama")
    from langchain_core.messages import HumanMessage, AIMessage
    from PIL import Image
    import matplotlib.pyplot as plt
    from langchain_ollama import OllamaLLM

# Define our simplified state schema
class ChatState(TypedDict):
    """Represents the state of our chatbot application."""
    messages: List[Dict[str, Any]]
    image_path: Optional[str]

# Initialize the LLM with a better balance of speed and quality
def get_llm():
    """Get the language model - trying a more balanced model"""
    try:
        # Mistral 7B is a good balance of speed and quality
        # If still too slow, you can try phi:mini instead
        return OllamaLLM(
            model="mistral:7b", 
            temperature=0.5,     # Lower temperature for more focused responses
            max_tokens=300       # Limit response length for speed
        )
    except Exception as e:
        # Fallback to phi:mini if mistral isn't available
        try:
            return OllamaLLM(model="phi:mini", temperature=0.5, max_tokens=300)
        except Exception:
            print(f"Error initializing Ollama: {str(e)}")
            print("Please make sure Ollama is installed and running.")
            print("Install from: https://ollama.ai/")
            print("After installing, run: 'ollama pull mistral:7b' or 'ollama pull phi:mini'")
            sys.exit(1)

# Simplified chatbot function - single model call for speed
def process_message(user_message: str, image_path: Optional[str] = None, chat_history: List[Dict] = None):
    """Process user input and return chatbot response."""
    if chat_history is None:
        chat_history = []
    
    # Add user message to chat history
    chat_history.append({"role": "human", "content": user_message})
    
    llm = get_llm()
    
    # Handle image information if present
    image_info = ""
    if image_path and Path(image_path).exists():
        image_info = f"\nThe user has also shared an image of {Path(image_path).name}. Consider this in your response."
    
    # Extract recent conversation context (just the last exchange to keep it fast)
    recent_context = ""
    if len(chat_history) > 2:
        recent_context = f"Previous message: {chat_history[-3]['content']}\nPrevious response: {chat_history[-2]['content']}\n"
    
    # Prepare the prompt - optimized for concise, specific answers
    full_prompt = f"""
    You are a recommendation assistant that provides clear, specific, and concise answers.
    
    {recent_context}
    User message: "{user_message}"{image_info}
    
    Your task:
    1. Provide 2-3 specific recommendations based on the user's request
    2. Be direct and get straight to the point - avoid unnecessary explanations 
    3. If the user asks for a list, provide a numbered list without long introductions
    4. End with a short, simple follow-up question
    
    Keep your entire response under 150 words. Be helpful but concise.
    """
    
    # Get the response from the LLM
    response = llm.invoke(full_prompt)
    
    # Add the AI response to the messages
    chat_history.append({"role": "ai", "content": response})
    
    return response, chat_history

# Display image if provided
def display_image(image_path: str):
    """Display the image using matplotlib."""
    if not image_path or not Path(image_path).exists():
        return
    
    try:
        img = Image.open(image_path)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Could not display image: {str(e)}")

# Main program
def main():
    """Main function to run the chatbot."""
    print("Optimized Recommendation Chatbot")
    print("Type 'exit' to quit")
    print("To share an image, type 'image:/path/to/image.jpg' after your message\n")
    
    chat_history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        
        image_path = None
        if "image:" in user_input:
            # Parse image path
            parts = user_input.split("image:")
            user_input = parts[0].strip()
            image_path = parts[1].strip()
            print(f"Image received: {image_path}")
            display_image(image_path)
        
        try:
            print("Chatbot is thinking...", end="", flush=True)
            response, chat_history = process_message(user_input, image_path, chat_history)
            print("\r" + " " * 20 + "\r", end="")  # Clear the "thinking" message
            print(f"Chatbot: {response}")
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Chatbot: I'm sorry, I encountered an error processing your request. Please try again.")

if __name__ == "__main__":
    main()
