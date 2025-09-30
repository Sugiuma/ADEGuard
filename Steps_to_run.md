https://drive.google.com/drive/folders/1pnQJfxpMILCO2r7FUH3wotmoJ6Y2vE4F?usp=sharing




Perfect! Here’s a **ready-to-use `inference.py`** that reads a CSV (`input.csv`) with a column `symptom_text`, runs BioBERT NER, post-processes entities, and outputs a CSV (`predicted_entities.csv`) with predicted ADE and DRUG columns.

---

```python
# inference.py

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import re
import pandas as pd
import argparse

# -----------------------------
# 1️⃣ Load model & tokenizer
# -----------------------------
model_path = "biobert-ner-final"  # update with your model folder path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -----------------------------
# 2️⃣ Post-processing dictionary
# -----------------------------
POSTPROCESS_DICT = {
    "DRUG": {"pfizer", "moderna", "astrazeneca", "covaxin",
             "janssen", "johnson", "johnson and johnson", "biontech"},
    "ADE": {"fever", "headache", "dizziness", "nausea",
            "rash", "fatigue", "chills", "itching", "sweating",
            "chest pain"}
}

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def postprocess_entities(text, entities):
    new_entities = {"DRUG": list(entities["DRUG"]), "ADE": list(entities["ADE"])}
    text_norm = normalize(text)

    for ent_type, vocab in POSTPROCESS_DICT.items():
        for word in vocab:
            word_norm = normalize(word)
            if word_norm in text_norm and not any(word_norm in normalize(e) for e in new_entities[ent_type]):
                new_entities[ent_type].append(word)
    return new_entities

def clean_entities(entities):
    cleaned = {"DRUG": [], "ADE": []}

    for ade in entities.get("ADE", []):
        ade = ade.strip("., ")
        if ade and ade.lower() not in ["and", "reported", "later", "severe"]:
            cleaned["ADE"].append(ade)

    for drug in entities.get("DRUG", []):
        drug = re.sub(r"\band\b.*", "", drug)
        drug = drug.strip("., ")
        if re.search(r"[A-Z]", drug) and len(drug.split()) <= 5:
            cleaned["DRUG"].append(drug)

    return cleaned

# -----------------------------
# 3️⃣ Predict function
# -----------------------------
def predict_entities(sentences):
    id2label = {0:"B-ADE", 1:"B-DRUG", 2:"I-ADE", 3:"I-DRUG", 4:"O"}
    results = []

    for sent in sentences:
        # Tokenize
        tokens = re.findall(r"\w+|[^\w\s]", sent)
        encoded = tokenizer(tokens, is_split_into_words=True,
                            truncation=True, max_length=512, return_tensors=None)
        inputs = {k: torch.tensor([v]).to(device) for k, v in encoded.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

        # Align tokens
        word_ids = encoded.word_ids(batch_index=0)
        pred_labels = []
        prev_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == prev_word_idx:
                continue
            pred_labels.append((tokens[word_idx], predictions[idx]))
            prev_word_idx = word_idx

        pred_labels_named = [(word, id2label[label]) for word, label in pred_labels]

        # Merge contiguous entities
        entities = {"DRUG": [], "ADE": []}
        current_entity = None
        current_words = []

        for word, label in pred_labels_named:
            if label in ["B-DRUG", "I-DRUG"]:
                if current_entity == "DRUG":
                    current_words.append(word)
                else:
                    if current_entity and current_words:
                        entities[current_entity].append(" ".join(current_words))
                    current_entity = "DRUG"
                    current_words = [word]
            elif label in ["B-ADE", "I-ADE"]:
                if current_entity == "ADE":
                    current_words.append(word)
                else:
                    if current_entity and current_words:
                        entities[current_entity].append(" ".join(current_words))
                    current_entity = "ADE"
                    current_words = [word]
            else:
                if current_entity and current_words:
                    entities[current_entity].append(" ".join(current_words))
                current_entity = None
                current_words = []

        if current_entity and current_words:
            entities[current_entity].append(" ".join(current_words))

        # Post-process
        entities_clean = clean_entities(entities)
        entities_post = postprocess_entities(sent, entities_clean)

        results.append(entities_post)

    return results

# -----------------------------
# 4️⃣ Main CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioBERT NER Inference for ADE/DRUG")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file with 'symptom_text' column")
    parser.add_argument("--output_csv", type=str, default="predicted_entities.csv", help="Path to save output CSV")
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.input_csv)
    if "symptom_text" not in df.columns:
        raise ValueError("Input CSV must contain 'symptom_text' column")

    # Run predictions
    entities_list = predict_entities(df["symptom_text"].astype(str).tolist())

    # Add predictions to DataFrame
    df["predicted_DRUG"] = [e["DRUG"] for e in entities_list]
    df["predicted_ADE"] = [e["ADE"] for e in entities_list]

    # Save output CSV
    df.to_csv(args.output_csv, index=False)
    print(f"✅ Predictions saved to {args.output_csv}")
```

---

## **5️⃣ How to Run**

1. Ensure dependencies:

```bash
pip install torch transformers pandas regex
```

2. Run inference on a CSV:

```bash
python inference.py --input_csv input.csv --output_csv predicted_entities.csv
```

* **`input.csv`** must have a column named `symptom_text`.
* The script outputs **`predicted_entities.csv`** with two new columns: `predicted_DRUG` and `predicted_ADE`.

---

This setup is **ready for batch inference** and can be integrated with your Streamlit UI.

If you want, I can also **modify it to handle large datasets efficiently in batches** so it won’t run out of memory on big VAERS files. Do you want me to do that?
