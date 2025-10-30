import joblib

vectorizer = joblib.load("tfidf_vectorizer_sentiment.pkl")
clf = joblib.load("svm_sentiment_classifier.pkl")

def classify_sentiment(text):
    vec = vectorizer.transform([text])
    pred = clf.predict(vec)[0]
    return pred

while True:
    txt = input("\nEnter a sentence (or 'exit'): ")
    if txt.lower() == 'exit':
        break
    print(f"→ Sentiment: {classify_sentiment(txt).upper()}")
