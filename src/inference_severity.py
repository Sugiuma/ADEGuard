from transformers import pipeline
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from config import C_MODEL_PATH

tokenizer = AutoTokenizer.from_pretrained(C_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(C_MODEL_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# 1️⃣ Label Mapping
# -----------------------------
id2label = {0: "Severe", 1: "Moderate", 2: "Mild"}

# -----------------------------
# 2️⃣ Load pipeline with your trained model
# -----------------------------
clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    return_all_scores=True
)

# -----------------------------
# 3️⃣ Severity Prediction
# -----------------------------
def predict_severity(text, severe_threshold=0.65, neutral_threshold=0.45):
    preds = clf(text)[0]  # list of dicts: [{'label': 'LABEL_0', 'score': 0.9}, ...]
    probs = np.array([p["score"] for p in preds])
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    # --- Heuristic calibration ---
    # If model is very confident but text is neutral, we dampen "Severe" predictions
    text_lower = text.lower()
    severe_keywords = ["hospital", "critical", "death", "severe", "intensive care","extreme", "intense", "life threatening"]
    mild_keywords = ["mild", "slight", "minor", "improved","light", "no immediate side effects"]

    # Check if text has any severe/mild indicators
    has_severe_kw = any(kw in text_lower for kw in severe_keywords)
    has_mild_kw = any(kw in text_lower for kw in mild_keywords)

    # Default label from model
    label = id2label.get(pred_idx, "Unknown")

    # --- Confidence-based override rules ---
    if label == "Severe" and not has_severe_kw and confidence < severe_threshold:
        # downgrade to Moderate if no severe signal
        label = "Moderate"
    elif label == "Mild" and not has_mild_kw and confidence < neutral_threshold:
        # neutral mild prediction with low confidence → Moderate
        label = "Moderate"

    # Prepare readable output
    prob_dict = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    return label, confidence, prob_dict

# -----------------------------
# 4️⃣ Example texts
# -----------------------------
examples = [
    "Patient developed high fever and severe headache after vaccination.",
    "Mild pain in the arm for one day.",
    "Critical condition and hospitalized due to allergic reaction.",
    "Slight fatigue for two days, now recovered."
]

# -----------------------------
# 5️⃣ Run predictions
# -----------------------------
for text in examples:
    label, conf, probs = predict_severity(text)
    print(f"🩸 Text: {text}")
    print(f"→ Predicted: {label} (confidence: {conf:.3f})")
    print(f"   Probabilities: {probs}\n")
