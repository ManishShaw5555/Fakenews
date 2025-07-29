# app.py

import streamlit as st
import joblib
from newspaper import Article

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to extract article text from URL
def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error fetching article: {str(e)}"

# Fake news detection pipeline
def detect_fake_news(text_or_url):
    if text_or_url.startswith("http"):
        article_text = extract_text_from_url(text_or_url)
        if article_text.startswith("Error"):
            return article_text
    else:
        article_text = text_or_url

    vec = vectorizer.transform([article_text])
    pred = model.predict(vec)[0]
    prob = model.decision_function(vec)[0]
    result = "✅ Real News" if pred == 1 else "🚫 Fake News"
    return f"{result} (Confidence: {round(abs(prob), 2)})"

# Streamlit UI
st.set_page_config(page_title="📰 FactShield", layout="centered")
st.title("📰 FactShield: Fake News Detector")
st.markdown("Paste a **news article** or a **URL**, and we'll predict whether it's real or fake.")

user_input = st.text_area("Enter article text or URL:", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text or a URL.")
    else:
        result = detect_fake_news(user_input)
        st.success(result)
