import joblib

# Load saved model and vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")
clf = joblib.load("svm_spam_classifier.pkl")

# Function to predict
def classify_message(message):
    message_vec = vectorizer.transform([message])
    prediction = clf.predict(message_vec)[0]
    return prediction

# Test
while True:
    msg = input("\nEnter message (or 'exit'): ")
    if msg.lower() == 'exit':
        break
    label = classify_message(msg)
    print(f"→ Prediction: {label.upper()}")
