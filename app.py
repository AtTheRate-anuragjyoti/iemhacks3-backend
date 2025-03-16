import os
from pathlib import Path
from flask import Flask, request, jsonify
from gpt4all import GPT4All
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
import requests

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure models directory exists
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Define model path
MODEL_PATH = MODEL_DIR / "gpt4all-lora-quantized.bin"

# Download model if it doesn't exist
if not MODEL_PATH.exists():
    try:
        url = "https://gpt4all.io/models/gpt4all-lora-quantized.bin"
        response = requests.get(url, stream=True)
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info("Model downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        raise

# Load GPT4All Model
try:
    model = GPT4All(str(MODEL_PATH))
    logger.info("GPT4All model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load GPT4All model: {str(e)}")
    raise

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "")
        response = model.generate(user_message)
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        text = data.get("text", "")
        sentiment = analyzer.polarity_scores(text)
        return jsonify(sentiment)
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

# Production Deployment
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))