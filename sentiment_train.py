import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# 1. Load dataset
df = pd.read_csv("IMDB Dataset.csv")
print(df.head())

# 2. Clean
df['review'] = df['review'].astype(str).str.lower()
df['sentiment'] = df['sentiment'].str.lower()

# 3. Split
X = df['review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. TF-IDF + SVM
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LinearSVC()
clf.fit(X_train_vec, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 6. Save
joblib.dump(vectorizer, "tfidf_vectorizer_sentiment.pkl")
joblib.dump(clf, "svm_sentiment_classifier.pkl")
