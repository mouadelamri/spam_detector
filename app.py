import streamlit as st
import joblib

# Load l-model mkhzen
model = joblib.load('spam_model.pkl')
tfidf = joblib.load('vectorizer.pkl')

st.title("🛡️ Spam Detector Pro")
st.write("Write a message below to check if it's Spam or Ham.")

user_input = st.text_area("Enter Message:")

if st.button("Predict"):
    if user_input:
        data = tfidf.transform([user_input])
        prediction = model.predict(data)[0]
        
        if prediction == 'spam':
            st.error(f"🚨 This message is: {prediction.upper()}")
        else:
            st.success(f"✅ This message is: {prediction.upper()}")
    else:
        st.warning("Please enter a message.")