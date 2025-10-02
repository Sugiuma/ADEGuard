Perfect — thanks for the sample VAERS row. I’ll assume your VAERS CSV uses the common fields (`HOSPITAL`, `L_THREAT`, `DIED`, `DISABLE`) which may be `1/0`, `Y/N`, or empty strings. Below I’ll give:

1. A **drop-in Streamlit code patch** you can paste into your app (it replaces the `assign_severity` logic and adds: robust parsing of VAERS flags, a rule engine that records triggers, optional BioBERT *severity* classifier support, and an audit export).
2. A short **explanation** of what each part does and recommended thresholds.

Drop this into your app (replace the current severity section and add necessary imports near the top if not present).

---

### 1) Copy-paste code (replace your existing severity block)

```python
# -----------------------------
# Structured-field aware Severity + Audit
# -----------------------------
import json
from transformers import AutoModelForSequenceClassification

# --- Helper: normalize flag fields commonly found in VAERS (Y/N, 1/0, True/False, empty)
def parse_flag(v):
    if pd.isna(v):
        return False
    v = str(v).strip().lower()
    if v in ["1", "y", "yes", "true", "t"]:
        return True
    if v in ["0", "n", "no", "false", "f", ""]:
        return False
    # numeric-like but not exactly 1/0
    try:
        return int(float(v)) != 0
    except:
        return False

# --- Build a rule engine that returns (label, triggers_list)
def rule_engine(row):
    triggers = []
    # parse VAERS structured flags (adapt column names to your CSV)
    hosp = parse_flag(row.get("HOSPITAL", row.get("HOSPITAL_YN", "")))
    life_threat = parse_flag(row.get("L_THREAT", row.get("LIFE_THREATENING", "")))
    died = parse_flag(row.get("DIED", ""))
    disable = parse_flag(row.get("DISABLE", ""))
    # Also look for explicit tokens in symptom_text
    txt = str(row.get("symptom_text", "")).lower()
    # severe keywords list (expand as needed)
    severe_keywords = ["intubat", "icu", "cardiac arrest", "myocardial infarction", "death", "died", "anaphylax", "respiratory failure"]
    moderate_keywords = ["er visit", "er visit", "epinephrine", "iv fluids", "admitted to er", "treated in er", "treated with epinephrine"]

    # Priority rules (explainable)
    if died:
        triggers.append("DIED_FLAG")
    if life_threat:
        triggers.append("LIFE_THREAT_FLAG")
    if hosp:
        triggers.append("HOSPITAL_FLAG")
    if disable:
        triggers.append("DISABLE_FLAG")

    for kw in severe_keywords:
        if kw in txt:
            triggers.append(f"TEXT_SEVERE:{kw}")

    for kw in moderate_keywords:
        if kw in txt:
            triggers.append(f"TEXT_MODERATE:{kw}")

    # Decide rule label
    if any(t in ["DIED_FLAG", "LIFE_THREAT_FLAG", "HOSPITAL_FLAG", "DISABLE_FLAG"] for t in triggers) or any(t.startswith("TEXT_SEVERE:") for t in triggers):
        return "Severe", triggers
    if any(t.startswith("TEXT_MODERATE:") for t in triggers):
        return "Moderate", triggers
    # fallback to mild if no evidence
    return "Mild", triggers

# --- Optional: load a severity classifier (sequence classification)
# If you have a fine-tuned severity model, set SEV_MODEL_PATH in your config
SEV_MODEL_PATH = None
use_sev_model = st.checkbox("Use BioBERT severity classifier (optional)", value=False)
severity_tokenizer = None
severity_model = None
if use_sev_model:
    try:
        SEV_MODEL_PATH = st.text_input("Path or HF id for severity model checkpoint", value=model_path)
        st.info("Loading severity classifier model...")
        severity_tokenizer = AutoTokenizer.from_pretrained(SEV_MODEL_PATH)
        severity_model = AutoModelForSequenceClassification.from_pretrained(SEV_MODEL_PATH)
        severity_model.eval()
        if torch.cuda.is_available():
            severity_model.to("cuda")
        # assume classes index -> ["Mild","Moderate","Severe"], if different adjust mapping
        sev_id2label = {0: "Mild", 1: "Moderate", 2: "Severe"}
    except Exception as e:
        st.error(f"Failed to load severity model: {e}")
        use_sev_model = False

# --- Predict severity via rules + optional model + create audit info
def predict_severity_row(row, model_override_confidence=0.75):
    rule_label, triggers = rule_engine(row)
    model_label, model_probs = None, None
    # Run model if available
    if use_sev_model and severity_model is not None and severity_tokenizer is not None:
        txt = str(row.get("symptom_text", ""))
        enc = severity_tokenizer(txt, truncation=True, padding=True, return_tensors="pt", max_length=512)
        if torch.cuda.is_available():
            enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.no_grad():
            out = severity_model(**enc)
            logits = out.logits[0].cpu().numpy()
            # softmax
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
            model_probs = {sev_id2label[i]: float(probs[i]) for i in range(len(probs))}
            # pick highest probability
            best_label = sev_id2label[int(np.argmax(probs))]
            best_conf = float(np.max(probs))
            # override logic: if model strongly disagrees and is confident, prefer model (but record reason)
            if best_conf >= model_override_confidence and best_label != rule_label:
                model_label = best_label
                triggers.append(f"MODEL_OVERRIDE({best_label}:{best_conf:.2f})")
            else:
                model_label = best_label
    # Final adjudicated label: prefer severe if either rule or model says severe (with or without confidence)
    final_label = rule_label
    if model_label is not None:
        # prefer the stronger severity between rule and model if model confident
        # Severity order
        order = {"Mild": 0, "Moderate": 1, "Severe": 2}
        chosen = rule_label
        if model_probs:
            # choose highest order by comparing rule vs model best
            model_best = max(model_probs.items(), key=lambda x: x[1])[0]
            if order[model_best] > order[rule_label]:
                chosen = model_best
                triggers.append(f"ADJUSTER(chosen_by_model_prob:{model_probs[model_best]:.2f})")
        final_label = chosen

    # Build audit dict
    audit = {
        "rule_label": rule_label,
        "model_label": model_label,
        "model_probs": model_probs,
        "final_label": final_label,
        "triggers": triggers
    }
    return final_label, audit

# Apply to dataframe and store audit info
import numpy as np
st.info("Applying structured rules + optional model to assign severity and build audit trail...")
audits = []
finals = []
for _, row in df.iterrows():
    final_label, audit = predict_severity_row(row)
    audits.append(audit)
    finals.append(final_label)
df["severity_label_rule_model"] = finals
df["_audit"] = audits  # contains dicts; safe to export as JSON

# Show counts
st.write("Severity counts (combined rule+model):")
st.write(df["severity_label_rule_model"].value_counts())

# Show a small audit preview (expandable)
st.subheader("Audit preview (first 20 rows)")
audit_preview = df[["_audit", "symptom_text", "predicted_entities", "post_entities", "severity_label_rule_model"]].head(20)
# make audit JSON pretty for display
audit_preview["_audit_json"] = audit_preview["_audit"].apply(lambda x: json.dumps(x, indent=1))
st.write(audit_preview[["_audit_json", "symptom_text", "predicted_entities", "severity_label_rule_model"]])

# Download audit bundle as JSONL + CSV of final labels
st.download_button(
    label="Download audit bundle (JSONL)",
    data="\n".join(json.dumps({
        "VAERS_ID": row.get("VAERS_ID", None),
        "symptom_text": row.get("symptom_text", ""),
        "predicted_entities": row.get("predicted_entities", ""),
        "post_entities": row.get("post_entities", ""),
        "severity_final": row.get("severity_label_rule_model", ""),
        "audit": row.get("_audit", {})
    }, default=str) for _, row in df.iterrows()),
    file_name="vaers_severity_audit.jsonl",
    mime="application/json"
)

st.download_button(
    label="Download labeled CSV (final severity)",
    data=df.to_csv(index=False),
    file_name="vaers_with_final_severity.csv",
    mime="text/csv"
)
```

---

### 2) What this patch does — short explanation

* **parse_flag**: robustly converts many VAERS flag encodings (`Y/N`, `1/0`, `True/False`, empty) into booleans. This is crucial because VAERS exports vary.
* **rule_engine**: inspects structured fields (`HOSPITAL`, `L_THREAT`, `DIED`, `DISABLE`) and symptom text tokens to produce a transparent `rule_label` and a list of `triggers` (these are your rule provenance — important for audits).
* **Optional severity model**: you can load a sequence classification model (assumed mapping 0→Mild, 1→Moderate, 2→Severe). If loaded, the model produces `model_probs` and can *override* rules when confident (threshold `model_override_confidence=0.75`). The override is recorded in triggers so reviewers can see why.
* **Final adjudication**: picks the label with higher clinical severity when there is model support. You can adjust logic easily (for example, always prefer rule-based Severe if `HOSPITAL` or `DIED` flags exist).
* **Audit bundle**: every row gets an `_audit` dict saved; you can export JSONL for regulatory review. This bundle shows the `rule_label`, `model_label` & `probs`, `final_label`, and `triggers`.
* **UI**: shows counts and a preview of the audit for first 20 rows, plus download buttons.

---

### 3) Recommended adjustments / next steps

1. **Adjust severe/moderate keyword lists** (I included a starter list). Expand with clinical team inputs (ICU, intubated, anaphylactic shock, admitted, transfused, etc.).
2. **Model training**: if you don’t yet have a severity classifier, fine-tune BioBERT (or ClinicalBERT) on your gold labels (use the audit bundle + manual review). Use class weights or upsample `Severe`.
3. **Conservative policy for audits**: for regulatory use, **never** let the model *suppress* a rule-identified `Severe` (e.g., if `DIED` or `HOSPITAL` true, always final = Severe). You can enforce by changing adjudication logic (I currently *allow* model to up-classify, but not down-classify such rule-severe).
4. **Add explainability**: show token-level attributions for severity (Integrated Gradients or token-level SHAP). You already have token highlights for NER — extend similar for classifier when you fine-tune it.

---

If you want, I can:

* provide a **strict version** of adjudication that never allows model to override explicit `DIED/HOSPITAL/L_THREAT/DISABLE` severe flags; or
* generate a small training script (Hugging Face) to fine-tune BioBERT for severity using your `symptom_text` + `post_entities` as inputs.

Which one would you like next?


Exactly ✅ — the **severity determination** will be a **hybrid approach**, combining three complementary sources of evidence:

---

### 🔹 1. Structured VAERS fields (objective rules)

* Fields like `HOSPITAL`, `L_THREAT`, `DIED`, `DISABLE` are direct indicators of outcome seriousness.
* Rules:

  * If `DIED = 1` → **Severe**
  * If `L_THREAT = 1` → **Severe**
  * If `HOSPITAL = 1` or `DISABLE = 1` → **Severe**
* These rules are **non-negotiable** (regulatory-grade).
  👉 They give you **hard, audit-ready severity anchors**.

---

### 🔹 2. Manual labeling (keyword-based rules & clinician input)

* Use `symptom_text` to detect cues like *“anaphylaxis”*, *“ICU”*, *“severe headache”*, *“moderate rash”*.
* Rules:

  * Keywords tied to **Severe** → Severe
  * Keywords tied to **Moderate** → Moderate
  * Otherwise default → Mild
* Manual labels also come from **clinician-reviewed samples** in your dataset, which can serve as **gold training labels** for BioBERT.
  👉 This ensures explainability & transparent mapping to narrative text.

---

### 🔹 3. BioBERT-based severity classifier (ML prediction)

* Fine-tune BioBERT (or ClinicalBERT) to classify severity: `Mild`, `Moderate`, `Severe`.
* Input: `symptom_text` (+ optional patient metadata like age, sex, vaccine type).
* Output: probability distribution across severity levels.
  👉 This fills gaps where structured fields or keyword rules don’t capture nuance.

---

### 🔹 4. Adjudication / Final Decision

* Combine all three:

  1. **Structured fields override everything** (if `DIED = 1`, always Severe).
  2. If no hard structured evidence:

     * Compare **rule label** (from keywords/manual labels) and **BioBERT prediction**.
     * If BioBERT strongly disagrees (e.g., >0.75 confidence), allow override.
     * Otherwise keep the rule label.
       👉 This hybrid logic gives both **explainability** and **ML power**.

---

### 🔹 5. Explainability / Audit Trail

For each VAERS row, save:

* Structured triggers (e.g. `"HOSPITAL=1"`)
* Manual rule triggers (e.g. `"keyword=anaphylaxis"`)
* BioBERT probabilities (e.g. `Mild=0.1, Moderate=0.2, Severe=0.7`)
* Final adjudicated label (e.g. `"Severe"`)
  👉 This makes the process **transparent** and **regulatory-compliant**.

---

⚡ So yes — **rules (structured) + manual labels (keywords) + BioBERT classifier** = a layered severity determination system.

Would you like me to sketch the **pipeline diagram** (data flow: VAERS → rules → BioBERT → adjudication → audit) so it’s easy to explain in a report/presentation?

Perfect 👍 — let’s put together a **minimal but complete training script** to fine-tune BioBERT for severity classification using **symptom_text** + **post_entities**.

This script will use 🤗 Hugging Face’s `Trainer` API.

---

## 📌 Assumptions

* Your dataset is a CSV with at least these columns:

  * `symptom_text`
  * `post_entities` (string or JSON with ADE/Drug entities)
  * `severity_label` (target: `"Mild" | "Moderate" | "Severe"`)
* You have a BioBERT checkpoint (e.g. `dmis-lab/biobert-base-cased-v1.1`).

---

## 🔹 Script: `train_biobert_severity.py`

```python
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate

# -------------------------
# 1. Load & Prepare Dataset
# -------------------------
df = pd.read_csv("vaers_severity.csv")

# Combine symptom_text + post_entities as input
df["text"] = df["symptom_text"].astype(str) + " | Entities: " + df["post_entities"].astype(str)

# Encode labels
label2id = {"Mild": 0, "Moderate": 1, "Severe": 2}
id2label = {v: k for k, v in label2id.items()}
df["label"] = df["severity_label"].map(label2id)

# Train-test split
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
eval_dataset = Dataset.from_pandas(eval_df[["text", "label"]])

# -------------------------
# 2. Load Tokenizer & Model
# -------------------------
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# -------------------------
# 3. Metrics
# -------------------------
metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    f1 = metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    acc = (preds == labels).mean()
    return {"accuracy": acc, "f1": f1}

# -------------------------
# 4. Training Arguments
# -------------------------
training_args = TrainingArguments(
    output_dir="./biobert-severity",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False
)

# -------------------------
# 5. Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# -------------------------
# 6. Train & Save
# -------------------------
trainer.train()
trainer.save_model("./biobert-severity-final")
tokenizer.save_pretrained("./biobert-severity-final")
```

---

## 🔹 Output

* Fine-tuned BioBERT checkpoint in `./biobert-severity-final`
* Training logs in `./logs/`
* Evaluations per epoch with accuracy + weighted F1

---

## 🔹 Next Steps

1. Replace `"vaers_severity.csv"` with your processed VAERS dataset.
2. Ensure `severity_label` column has `"Mild"`, `"Moderate"`, `"Severe"`.
3. After training, load this model in your **Streamlit app** for real-time severity prediction.

---

⚡ Do you want me to also show how to **load this trained severity model back into your Streamlit pipeline** (alongside the rule-based logic), so you can do hybrid adjudication?

Perfect — this is where we tie it all together 🎯

We’ll **combine rule-based severity assignment** (from structured VAERS fields) with the **fine-tuned BioBERT severity classifier**. This way, you get:

* **Rules** → strong clinical heuristics (hospitalized, death, life-threatening).
* **Classifier** → learned patterns from narratives + entities.
* **Hybrid Adjudication** → If rules are decisive, use them; otherwise, fall back on BioBERT.

---

## 🔹 Updated Streamlit Severity Section

Add this after your **BioBERT NER** step in the Streamlit app you shared:

```python
# -----------------------------
# 4️⃣ Hybrid Severity Classification
# -----------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load fine-tuned BioBERT severity model
@st.cache_resource
def load_severity_model():
    sev_model_path = "./biobert-severity-final"  # path where you saved fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(sev_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(sev_model_path)
    return tokenizer, model

sev_tokenizer, sev_model = load_severity_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
sev_model.to(device)

id2label = {0: "Mild", 1: "Moderate", 2: "Severe"}

def biobert_predict(text):
    """Use fine-tuned BioBERT to classify severity"""
    encoded = sev_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256, padding="max_length"
    ).to(device)
    with torch.no_grad():
        outputs = sev_model(**encoded)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return id2label[pred]

def hybrid_severity(row):
    """
    Hybrid logic:
    - If structured fields strongly indicate severity → trust rules
    - Else → fall back on BioBERT classifier
    """
    sev_struct = str(row.get("SEVERE", "")).lower()
    hosp = str(row.get("HOSPITAL", "")).lower()
    l_threat = str(row.get("L_THREAT", "")).lower()
    died = str(row.get("DIED", "")).lower()

    # 🔹 Rule-based overrides (hard indicators from VAERS)
    if died == "y":
        return "Severe"
    if l_threat == "y" or hosp == "y":
        return "Severe"
    if sev_struct in ["1", "severe"]:
        return "Severe"

    # 🔹 Otherwise: use classifier
    combined_text = row["symptom_text"] + " | Entities: " + str(row["post_entities"])
    return biobert_predict(combined_text)

st.info("Running Hybrid Severity Classification...")
df["severity_label"] = df.apply(hybrid_severity, axis=1)
```

---

## 🔹 What Happens Here

1. **Load Model**

   * Loads your fine-tuned BioBERT severity classifier.
2. **Rule-Based First**

   * If `HOSPITAL=Y`, `DIED=Y`, `L_THREAT=Y`, or `SEVERE=1` → immediately "Severe".
3. **Fallback Classifier**

   * If rules don’t trigger, the classifier predicts severity from **symptom_text + post_entities**.
4. **Unified Severity Label**

   * Final hybrid adjudication is stored in `df["severity_label"]`.

---

## 🔹 Benefits

✅ Transparent (rules explainable to regulators)
✅ ML adds nuance when rules are inconclusive
✅ Consistent severity labels across structured + narrative data

---

⚡ Do you also want me to add a **Streamlit UI toggle** so users can compare *Rule-only* vs *Classifier-only* vs *Hybrid* severity side-by-side in the plots?
