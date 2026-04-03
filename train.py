import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load & Clean Data
# Kaggle dataset fih latin-1 encoding w columns zaydine
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# 2. Vectorization (TF-IDF is better than CountVectorizer for GitHub projects)
tfidf = TfidfVectorizer(stop_words='english', lowercase=True)
X = tfidf.fit_transform(df['text'])
y = df['label']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Save Model & Vectorizer (Professional Practice)
# Bach n-asta3mlohom men ba3d f l-Interface (UI)
joblib.dump(model, 'spam_model.pkl')
joblib.dump(tfidf, 'vectorizer.pkl')
print("✅ Model and Vectorizer saved!")

# 6. Evaluate & Visualize
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Confusion Matrix Visual
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png') # Save for GitHub README
plt.show()

print("\nFull Report:\n", classification_report(y_test, y_pred))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove punctuation
    #remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
df['text'] = df['text'].apply(clean_text)