import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset (replace with real dataset for better results)
data = {
    "text": [
        "I love this product",
        "This is amazing",
        "Very bad experience",
        "I hate this",
        "It's okay, not great",
        "Average quality",
        "Excellent service",
        "Worst purchase ever",
        "Really happy with this",
        "Not worth the money",
        "It is fine",
        "Superb quality",
        "Terrible support",
        "Good but can improve",
        "I am satisfied"
    ],
    "sentiment": [
        "positive","positive","negative","negative","neutral",
        "neutral","positive","negative","positive","negative",
        "neutral","positive","negative","neutral","positive"
    ]
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")