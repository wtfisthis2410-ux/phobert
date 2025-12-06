# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer
from model_multihead import MultiHeadPhoBERT
import os

MODEL_DIR = os.environ.get("MODEL_DIR", "./distilphobert-checkpoint")
BASE_MODEL = os.environ.get("BASE_MODEL", "vinai/phobert-base")

# label lists must match training
BULLYING_LABELS = ["physical","verbal","sexual","social","cyber","none"]
SEVERITY_LABELS = ["low","medium","high","critical"]
EMOTION_LABELS = ["neutral","sad","angry","fear","happy","other"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)  # tokenizer saved in checkpoint

# load model (structure must match)
model = MultiHeadPhoBERT(BASE_MODEL, n_labels_bullying=len(BULLYING_LABELS),
                         n_labels_severity=len(SEVERITY_LABELS),
                         n_labels_emotion=len(EMOTION_LABELS))
# load weights
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "pytorch_model.bin"), map_location=device))
model.to(device)
model.eval()

app = Flask(__name__)
CORS(app)

def inference_text(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    enc = {k:v.to(device) for k,v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    import torch.nn.functional as F
    probs_b = F.softmax(out["logits_bullying"], dim=-1)[0].cpu().numpy()
    probs_s = F.softmax(out["logits_severity"], dim=-1)[0].cpu().numpy()
    probs_e = F.softmax(out["logits_emotion"], dim=-1)[0].cpu().numpy()

    return {
        "bullying_label": BULLYING_LABELS[int(probs_b.argmax())],
        "bullying_probs": probs_b.tolist(),
        "severity": SEVERITY_LABELS[int(probs_s.argmax())],
        "severity_probs": probs_s.tolist(),
        "emotion": EMOTION_LABELS[int(probs_e.argmax())],
        "emotion_probs": probs_e.tolist(),
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
