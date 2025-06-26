import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import io

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Knowledge base
corpus = [
    "Hello! How can I help you today?",
    "I am a smart assistant created for the CodTech internship.",
    "What is artificial intelligence?",
    "Artificial Intelligence is the simulation of human intelligence by machines.",
    "What is AI?",
    "AI stands for Artificial Intelligence.",
    "AI full form?",
    "Abbreviation of AI?",
    "Meaning of AI?",
    "I want the full form of AI.",
    "What is Python?",
    "Python is a powerful programming language used in web, AI, and data science.",
    "Tell me a joke.",
    "Why do programmers prefer dark mode? Because light attracts bugs!",
    "How are you?",
    "I'm just a program, but I'm functioning as expected!",
    "What can you do?",
    "I can answer simple questions using NLP and machine learning.",
    "Goodbye",
    "See you later! Have a great day!"
]

corpus_lower = [doc.lower() for doc in corpus]

# Custom rule-based logic before similarity
def rule_based_reply(user_input):
    user_input = user_input.lower()
    if "ai" in user_input and ("full" in user_input or "form" in user_input or "abbreviation" in user_input or "meaning" in user_input):
        return "AI stands for Artificial Intelligence."
    elif "joke" in user_input:
        return "Why do programmers prefer dark mode? Because light attracts bugs!"
    elif "how are you" in user_input:
        return "I'm just a program, but I'm functioning as expected!"
    elif "what can you do" in user_input:
        return "I can answer simple questions using NLP and machine learning."
    return None

# TF-IDF based response
def generate_response(user_input):
    rule_response = rule_based_reply(user_input)
    if rule_response:
        return rule_response
    corpus_with_input = corpus_lower + [user_input.lower()]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus_with_input)
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    index = similarity.argsort()[0][-1]
    score = similarity[0][index]
    return corpus[index] if score > 0.2 else "I'm sorry, I couldn't understand that. Can you rephrase?"

# Logging function with safe Unicode handling
def log_conversation(user_input, bot_response):
    with io.open("chat_log.txt", "a", encoding="utf-8") as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] You: {user_input}\n")
        log_file.write(f"[{timestamp}] CodTechBot: {bot_response}\n\n")  # Removed 🤖 to avoid errors

# Chat loop
print("CodTechBot 🤖: Hello! I'm your chatbot. Ask me anything. Type 'bye' to exit.")
while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "bye":
        print("CodTechBot 🤖: Goodbye! Stay safe.")
        break
    elif user_input.strip() == "":
        print("CodTechBot 🤖: Please enter something!")
    else:
        response = generate_response(user_input)
        print("CodTechBot 🤖:", response)
        log_conversation(user_input, response)
