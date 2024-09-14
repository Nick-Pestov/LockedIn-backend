from dotenv import load_dotenv
import os
import cohere

load_dotenv()

client = os.getenv("CLIENT")
modifier = os.getenv("MODIFIER")

chat_history = []

co = cohere.Client(client)

def preprocess(message):
    return modifier + message

def respond(message):
    if not message: return "Please enter a message."

    message = preprocess(message)

    # Get the response from the assistant
    response = co.chat(
        message,
        model="command",
        temperature=0.3,
        chat_history=chat_history
    ).text

    # Append the user and assistant messages to the chat history
    chat_history.append({
        "user_name": "User",
        "text": message
    })

    chat_history.append({
        "user_name": "Assistant",
        "text": response
    })

    return response