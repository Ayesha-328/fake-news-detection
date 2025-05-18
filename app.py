import streamlit as st
import joblib
import numpy as np

# Load your model and vectorizer
model = joblib.load("model.pkl")           # trained XGBoost model
vectorizer = joblib.load("vectorizer.pkl") # TF-IDF vectorizer

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("Enter a news article below to check if it's **Fake** or **Real**.")

user_input = st.text_area("📝 Paste News Article Here", height=300)

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input and make prediction
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        # Display result
        if prediction == 1:
            st.error("❌ This news article is **Fake**.")
        else:
            st.success("✅ This news article is **Real**.")
