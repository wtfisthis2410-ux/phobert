from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

HF_API_KEY = os.environ.get("HF_API_KEY")  # Lấy key từ .env hoặc biến môi trường
MODEL_ID = "funa21/phobert-finetuned-victsd-toxic-v2"

API_URL = "https://router.huggingface.co/inference"
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

def call_hf_api(text):
    payload = {
        "model": MODEL_ID,
        "input": text
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)

    # Trường hợp model chưa warm-up hoặc error từ HF
    try:
        return response.json()
    except:
        return {"error": "Invalid response from HuggingFace"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "no text provided"}), 400

    result = call_hf_api(text)
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    return {"status": "backend running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
