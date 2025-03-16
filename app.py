import os
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from gpt4all import GPT4All
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ✅ Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ✅ Model Directory
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ✅ Model Name (Let GPT4All Auto-Download)
MODEL_NAME = "Nous Hermes 2 Mistral DPO"

# ✅ Load GPT-4All Model (Auto-downloads if missing)
try:
    model = GPT4All(MODEL_NAME, model_path=str(MODEL_DIR))
    logger.info("✅ GPT-4All Model Loaded Successfully!")
except Exception as e:
    logger.error(f"❌ Failed to Load GPT-4All Model: {str(e)}")
    raise

# ✅ Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "")
        response = model.generate(user_message)
        return jsonify({"response": response})
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

# ✅ Production Deployment
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
