import spacy
import random
import json
import os
import numpy as np  # Import numpy for norm calculation

# Load the spaCy language model
try:
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    exit(1)

# Define intents as a JSON structure
intents = {
    "greetings": {
        "patterns": ["hello", "hi", "hey", "howdy", "what's up", "greetings"],
        "responses": ["Hello! How can I assist you?", "Hi there! What's on your mind?", "Hey! How can I help?"]
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "see you later", "take care"],
        "responses": ["Goodbye! Have a great day!", "See you later!", "Take care!"]
    },
    "thanks": {
        "patterns": ["thanks", "thank you", "much appreciated"],
        "responses": ["You're welcome!", "No problem!", "Happy to help!"]
    },
    "weather": {
        "patterns": ["what's the weather", "tell me the weather", "current weather", "weather forecast"],
        "responses": ["I'm not connected to a weather API, but I hope it's sunny where you are!", "It might be a good idea to check a weather app for accurate details."]
    },
    "default": {
        "responses": ["I'm sorry, I don't understand that.", "Can you rephrase that?", "I'm not sure how to help with that."]
    }
}

# Precompute vector representations for patterns
def preprocess_patterns():
    pattern_vectors = {}
    for intent, data in intents.items():
        pattern_vectors[intent] = [nlp(pattern).vector for pattern in data.get("patterns", [])]
    return pattern_vectors

# Precompute the pattern vectors for faster matching
pattern_vectors = preprocess_patterns()

# Store user info (in-memory session-like structure)
user_info = {}

# Find the best matching intent using spaCy vectors
def match_intent(user_input):
    user_doc = nlp(user_input.lower())  # Make input lowercase for better matching
    best_match = {"intent": "default", "score": 0}

    for intent, vectors in pattern_vectors.items():
        for pattern_vector in vectors:
            # Calculate cosine similarity using numpy
            similarity = np.dot(user_doc.vector, pattern_vector) / (np.linalg.norm(user_doc.vector) * np.linalg.norm(pattern_vector))
            if similarity > best_match["score"]:
                best_match = {"intent": intent, "score": similarity}

    return best_match["intent"]

# Get a response for a matched intent
def get_response(intent):
    return random.choice(intents[intent]["responses"])

# Log conversation along with the intent
def log_conversation(user_input, chatbot_response, intent):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "chat_log_spacy.txt")
    
    try:
        with open(log_file_path, "a") as log_file:
            log_file.write(f"User: {user_input}\n")
            log_file.write(f"Chatbot: {chatbot_response}\n")
            log_file.write(f"Intent: {intent}\n\n")
    except Exception as e:
        print(f"Error while logging the conversation: {e}")

# Ask for the user's name if not stored
def ask_name():
    if "name" not in user_info:
        user_info["name"] = input("What's your name? ").strip()
        print(f"Nice to meet you, {user_info['name']}!")

# Main chatbot function
def chatbot():
    print("Chatbot: Hello! I'm here to assist you. Type 'exit' to end the conversation.")

    # Ask for name only once at the beginning of the conversation
    ask_name()

    while True:
        try:
            user_input = input(f"You: ").strip()

            if user_input.lower() == "exit":
                print("Chatbot: Goodbye! Have a great day!")
                break
            
            # If the name is stored, include it in the greeting
            greeting = f"Hello {user_info['name']}! " if "name" in user_info else ""

            # Match intent and respond
            intent = match_intent(user_input)
            response = get_response(intent)
            print(f"Chatbot:{response}")

            # Log the conversation
            log_conversation(user_input, response, intent)
        except Exception as e:
            print(f"Error occurred while processing the input: {e}")

if __name__ == "__main__":
    try:
        chatbot()
    except Exception as e:
        print(f"Error in chatbot execution: {e}")
