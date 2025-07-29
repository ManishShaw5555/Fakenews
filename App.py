import gradio as gr
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

# Main pipeline function
def detect_fake_news(input_text_or_url):
    # If it's a URL
    if input_text_or_url.startswith("http"):
        article_text = extract_text_from_url(input_text_or_url)
        if article_text.startswith("Error"):
            return article_text
    else:
        article_text = input_text_or_url

    # Prediction
    vec = vectorizer.transform([article_text])
    pred = model.predict(vec)[0]
    prob = model.decision_function(vec)[0]
    result = "✅ Real News" if pred == 1 else "🚫 Fake News"
    return f"{result} (Confidence: {round(abs(prob), 2)})"

# Gradio UI
iface = gr.Interface(
    fn=detect_fake_news,
    inputs=gr.Textbox(lines=2, placeholder="Paste article URL or text here..."),
    outputs="text",
    title="📰 FactShield: Fake Article Detector",
    description="Paste a news article or a URL. AI will predict whether it's real or fake."
)

iface.launch()

