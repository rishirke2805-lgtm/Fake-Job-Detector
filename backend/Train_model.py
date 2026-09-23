import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # <-- Added import here at the top

# --------------------------------
# 1. Load processed dataset
# --------------------------------
df = pd.read_csv("dataset/processed_jobs.csv")
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# --------------------------------
# 2. Separate input and target
# --------------------------------
X = df["combined_text"]
y = df["fraudulent"]

# --------------------------------
# 3. Train/Test split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------
# 4. Convert text into TF-IDF
# --------------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("\nTF-IDF conversion completed!")
print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)

# --------------------------------
# 5. Train Logistic Regression
# --------------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
model.fit(X_train_tfidf, y_train)
print("\nModel training completed!")

# --------------------------------
# 6. Make predictions & Evaluate
# --------------------------------
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")
print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Genuine", "Fraudulent"]
))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------
# 7. Save the trained model & vectorizer
# --------------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved as model.pkl")
print("TF-IDF vectorizer saved as tfidf_vectorizer.pkl")
print("Phase 3 completed successfully!")