import joblib

# Load both models
spam_vectorizer = joblib.load("tfidf_vectorizer.pkl")
spam_clf = joblib.load("svm_spam_classifier.pkl")

sent_vectorizer = joblib.load("tfidf_vectorizer_sentiment.pkl")
sent_clf = joblib.load("svm_sentiment_classifier.pkl")

def analyze_email(text):
    """Classify email as Spam/Ham + (Positive/Negative if Ham)"""
    # Step 1: Spam check
    spam_vec = spam_vectorizer.transform([text])
    spam_pred = spam_clf.predict(spam_vec)[0]

    if spam_pred == "spam":
        return {"spam_ham": "SPAM", "sentiment": None}

    # Step 2: Sentiment check (for Ham only)
    sent_vec = sent_vectorizer.transform([text])
    sent_pred = sent_clf.predict(sent_vec)[0]

    return {"spam_ham": "HAM", "sentiment": sent_pred.upper()}

# 🔍 Test interactively
if __name__ == "__main__":
    while True:
        msg = input("\nEnter email text (or 'exit'): ")
        if msg.lower() == "exit":
            break
        result = analyze_email(msg)
        print("→ Spam/Ham:", result["spam_ham"])
        if result["sentiment"]:
            print("→ Sentiment:", result["sentiment"])
