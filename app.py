import os
import logging
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ✅ Load API Key from .env
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# ✅ Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ✅ Hugging Face API Details
MODEL_NAME = "Qwen/Qwen-QwQ-32B"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# ✅ Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "")

        response = requests.post(API_URL, headers=HEADERS, json={"inputs": user_message})
        result = response.json()
        generated_text = result[0]["generated_text"] if isinstance(result, list) else "Error: Invalid response"

        return jsonify({"response": generated_text})
    except Exception as e:
        logger.error(f"❌ Chat Error: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        text = data.get("text", "")
        sentiment = analyzer.polarity_scores(text)
        return jsonify(sentiment)
    except Exception as e:
        logger.error(f"❌ Sentiment Analysis Error: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

# ✅ Production Deployment with Gunicorn
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
