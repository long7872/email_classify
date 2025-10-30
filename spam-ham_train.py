import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load data (update path to where you downloaded it)
df = pd.read_csv("email.csv")  # adjust filename

# Clean up invalid rows
df = df[df['Category'].isin(['ham', 'spam'])]

# Inspect columns
print(df.columns)
print(df['Category'].value_counts())
print(df['Message'].value_counts())
# Suppose the text column is named 'text', label column is named 'label'

# 2. Pre-process & split
X = df['Message'].astype(str)
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Feature extraction
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train SVM
clf = LinearSVC()
clf.fit(X_train_vec, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Save model & vectorizer
import joblib
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(clf, "svm_spam_classifier.pkl")
