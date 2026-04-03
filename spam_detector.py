import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split # New
from sklearn.metrics import accuracy_score          # New
import re
import nltk
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove punctuation
    return text

# Use TF-IDF instead of CountVectorizer (more professional)
# vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

# 1. Load Data
data = pd.read_csv('data.csv')

# 2. Split into Training and Testing sets (Optional but professional)
# This helps you check if the model actually works on new data
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# 3. Vectorization
vectorizer = CountVectorizer(stop_words='english', lowercase=True)
X_train_vectorized = vectorizer.fit_transform(X_train)

# 4. Train Model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# 5. Check Performance
X_test_vectorized = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vectorized)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100}%")

# 6. Better Prediction Logic
def predict_spam(message):
    if not message.strip():
        return "Empty input"
    
    msg_vec = vectorizer.transform([message])
    # predict_proba shows you the percentage of certainty
    proba = model.predict_proba(msg_vec) 
    prediction = model.predict(msg_vec)
    
    return prediction[0]

# Try a real test
test_msg = "WIN CASH PRIZE NOW"
print(f"Result: {predict_spam(test_msg)}")