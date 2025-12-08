# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch.nn.functional as F

# dùng model fine-tuned trên HF
MODEL_NAME = os.environ.get("MODEL_NAME", "funa21/phobert-finetuned-victsd-toxic-v2")

# labels từ fine-tuned model (bạn có thể sửa theo model thật)
LABELS = ["physical","verbal","sexual","social","cyber","none"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# load tokenizer và model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

app = Flask(__name__)
CORS(app)

def inference_text(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    enc = {k:v.to(device) for k,v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    probs = F.softmax(out.logits, dim=-1)[0].cpu().numpy()
    return {
        "label": LABELS[int(probs.argmax())],
        "probs": probs.tolist()
    }

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text","")
    if not text:
        return jsonify({"error":"no text"}), 400
    res = inference_text(text)
    return jsonify(res)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"running"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
