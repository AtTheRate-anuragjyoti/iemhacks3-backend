import os
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from groq import Groq

# ✅ Load API Key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ✅ Enable CORS
CORS(app)

# ✅ Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

# ✅ Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "")

        # ✅ Send request to Groq API
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # Change to any available model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=256,
            top_p=1.0
        )

        # ✅ Extract response from Groq
        generated_text = response.choices[0].message.content

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

# ✅ Production Deployment
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))