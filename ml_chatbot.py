from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training Data
questions = [
    "hello",
    "hi",
    "good morning",
    "what is python",
    "tell me about python",
    "what is sql",
    "tell me about sql",
    "what is machine learning",
    "explain machine learning",
    "how to become data analyst",
    "data analyst skills",
    "what is power bi",
    "tell me about power bi"
]

labels = [
    "greeting",
    "greeting",
    "greeting",
    "python",
    "python",
    "sql",
    "sql",
    "ml",
    "ml",
    "career",
    "career",
    "powerbi",
    "powerbi"
]

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions)

# Train Model
model = MultinomialNB()
model.fit(X, labels)

print("=" * 50)
print("     MACHINE LEARNING CHATBOT")
print("=" * 50)

while True:

    user = input("\nYou: ").lower()

    if user == "bye":
        print("Bot: Goodbye!")
        break

    user_vector = vectorizer.transform([user])

    prediction = model.predict(user_vector)[0]

    if prediction == "greeting":
        print("Bot: Hello! How can I help you?")

    elif prediction == "python":
        print("Bot: Python is a powerful programming language used in AI, Data Science, and Web Development.")

    elif prediction == "sql":
        print("Bot: SQL is used to manage and retrieve data from databases.")

    elif prediction == "ml":
        print("Bot: Machine Learning enables computers to learn patterns from data.")

    elif prediction == "career":
        print("Bot: To become a Data Analyst, learn SQL, Python, Power BI, and Excel.")

    elif prediction == "powerbi":
        print("Bot: Power BI is a business intelligence tool used for dashboards and reporting.")