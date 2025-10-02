Perfect 😎 — let’s combine everything into **one unified pipeline**:

**Goal:**

1. **Row-level severity (Severe/Moderate/Mild) with Snorkel probabilities**
2. **Token-level ADE/DRUG BIO labels** automatically extracted from text + symptom columns
3. **Save CSV + JSONL** for BioBERT training

---

### 🔹 Full Unified Pipeline

```python
import pandas as pd
import re
import json
from snorkel.labeling import LabelingFunction, PandasLFApplier, LabelModel
from sklearn.model_selection import train_test_split

# -----------------------------
# 1️⃣ Load VAERS
# -----------------------------
df = pd.read_csv("vaers_data.csv")

# -----------------------------
# 2️⃣ ADE / DRUG dictionaries
# -----------------------------
ade_terms = ["pain","headache","fever","chills","insomnia","fatigue","nausea","vomiting","dizziness","rash"]
drug_terms = ["flu shot","fluzone","vaccine","acetaminophen","ibuprofen"]

# -----------------------------
# 3️⃣ Snorkel Labeling Functions
# -----------------------------
SEVERE, MODERATE, MILD, ABSTAIN = 0, 1, 2, -1
id2label = {0: "Severe", 1: "Moderate", 2: "Mild"}

def lf_died(row): return SEVERE if str(row.get("DIED","")).strip().upper() == "Y" else ABSTAIN
def lf_hospital(row): return SEVERE if str(row.get("HOSPITAL",0)) in ["1","Y","YES"] else ABSTAIN
def lf_l_threat(row): return SEVERE if str(row.get("L_THREAT","")).strip().upper() == "Y" else ABSTAIN
def lf_disable(row): return SEVERE if str(row.get("DISABLE","")).strip().upper() == "Y" else ABSTAIN
def lf_text_severe(row):
    text = str(row.get("SYMPTOM_TEXT","")).lower()
    return SEVERE if any(w in text for w in ["death","fatal","critical","severe","life threatening"]) else ABSTAIN
def lf_text_moderate(row):
    return MODERATE if "moderate" in str(row.get("SYMPTOM_TEXT","")).lower() else ABSTAIN
def lf_text_mild(row):
    return MILD if "mild" in str(row.get("SYMPTOM_TEXT","")).lower() else ABSTAIN

lfs = [
    LabelingFunction("lf_died", lf_died),
    LabelingFunction("lf_hospital", lf_hospital),
    LabelingFunction("lf_l_threat", lf_l_threat),
    LabelingFunction("lf_disable", lf_disable),
    LabelingFunction("lf_text_severe", lf_text_severe),
    LabelingFunction("lf_text_moderate", lf_text_moderate),
    LabelingFunction("lf_text_mild", lf_text_mild),
]

# -----------------------------
# 4️⃣ Apply LFs & Train Label Model
# -----------------------------
applier = PandasLFApplier(lfs)
L_train = applier.apply(df)

label_model = LabelModel(cardinality=3, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=42)

# Probabilities + hard labels
probs = label_model.predict_proba(L=L_train)
df["weak_label_prob_SEVERE"] = probs[:, SEVERE]
df["weak_label_prob_MODERATE"] = probs[:, MODERATE]
df["weak_label_prob_MILD"] = probs[:, MILD]
df["weak_label_id"] = label_model.predict(L=L_train)
df["weak_label"] = df["weak_label_id"].map(id2label)

# -----------------------------
# 5️⃣ Automatic ADE/DRUG weak labels
# -----------------------------
def extract_entity_spans(row):
    text = str(row.get("SYMPTOM_TEXT",""))
    spans = []

    # From SYMPTOM1..SYMPTOM5
    for col in ["SYMPTOM1","SYMPTOM2","SYMPTOM3","SYMPTOM4","SYMPTOM5"]:
        val = row.get(col,"")
        if pd.notna(val):
            val_lower = str(val).lower()
            for term in ade_terms:
                if term in val_lower:
                    for m in re.finditer(r"\b{}\b".format(re.escape(term)), text.lower()):
                        spans.append({"text": text[m.start():m.end()],
                                      "start": m.start(),
                                      "end": m.end(),
                                      "label": "ADE"})
            for term in drug_terms:
                if term in val_lower:
                    for m in re.finditer(r"\b{}\b".format(re.escape(term)), text.lower()):
                        spans.append({"text": text[m.start():m.end()],
                                      "start": m.start(),
                                      "end": m.end(),
                                      "label": "DRUG"})

    # From free text
    lower_text = text.lower()
    for term in ade_terms:
        for m in re.finditer(r"\b{}\b".format(re.escape(term)), lower_text):
            spans.append({"text": text[m.start():m.end()],
                          "start": m.start(),
                          "end": m.end(),
                          "label": "ADE"})
    for term in drug_terms:
        for m in re.finditer(r"\b{}\b".format(re.escape(term)), lower_text):
            spans.append({"text": text[m.start():m.end()],
                          "start": m.start(),
                          "end": m.end(),
                          "label": "DRUG"})

    return spans

df["weak_entity_spans"] = df.apply(extract_entity_spans, axis=1)

# -----------------------------
# 6️⃣ BIO conversion
# -----------------------------
def text_to_bio(text, entities):
    tokens = re.findall(r"\w+|[^\w\s]", text)
    labels = ["O"] * len(tokens)
    for ent in entities:
        ent_tokens = re.findall(r"\w+|[^\w\s]", ent["text"])
        for i in range(len(tokens)):
            if tokens[i:i+len(ent_tokens)] == ent_tokens:
                labels[i] = f"B-{ent['label']}"
                for j in range(1,len(ent_tokens)):
                    labels[i+j] = f"I-{ent['label']}"
                break
    return tokens, labels

bio_data = []
for _, row in df.iterrows():
    text = str(row.get("SYMPTOM_TEXT",""))
    entities = row.get("weak_entity_spans", [])
    tokens, labels = text_to_bio(text, entities)
    bio_data.append({
        "id": row["VAERS_ID"],
        "tokens": tokens,
        "labels": labels,
        "weak_label": row["weak_label"],
        "weak_label_probs": {
            "SEVERE": row["weak_label_prob_SEVERE"],
            "MODERATE": row["weak_label_prob_MODERATE"],
            "MILD": row["weak_label_prob_MILD"]
        }
    })

# -----------------------------
# 7️⃣ Train/Val/Test Split + Save
# -----------------------------
train, temp = train_test_split(bio_data, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

def save_jsonl(filename, data):
    with open(filename,"w") as f:
        for item in data:
            f.write(json.dumps(item)+"\n")

save_jsonl("train.jsonl", train)
save_jsonl("val.jsonl", val)
save_jsonl("test.jsonl", test)

# Save CSV with weak labels & probabilities
df.to_csv("vaers_final_pipeline.csv", index=False)

print("✅ Full pipeline complete: CSV + BIO JSONL ready for BioBERT training")
```

---

### 🔹 What You Get

1. **`vaers_final_pipeline.csv`**

   * All original VAERS columns
   * `weak_label` → Snorkel hard label (Severe/Moderate/Mild)
   * `weak_label_prob_SEVERE / MODERATE / MILD` → probabilistic soft labels
   * `weak_entity_spans` → automatically detected ADE/DRUG spans

2. **`train.jsonl / val.jsonl / test.jsonl`**

   * `tokens` + `labels` (B-ADE/I-ADE, B-DRUG/I-DRUG, O)
   * `weak_label` + `weak_label_probs` included per row → usable for joint training

---

This is now **fully automated**:

* No manual entity annotations needed
* Weak NER labels + Snorkel-based severity labels
* Ready for **BioBERT NER + severity classifier** training

---

If you want, I can also **show a snippet to load this JSONL directly in HuggingFace `datasets`** and prepare it for **multi-task BioBERT** (NER + severity regression/classification).

Do you want me to do that next?

Absolutely ✅ — your JSONL can be **loaded into Label Studio** for quick review and correction, with a few tweaks. Here’s how you can do it:

---

### 🔹 1️⃣ Modify JSONL for Label Studio

Label Studio expects each row to have:

* `text` → the full text to annotate
* `entities` → list of spans (with `start`, `end`, `label`)

Right now your `bio_data` JSONL has `tokens` + `labels`, which is **token-level**. To review in Label Studio, we need **span-level entities**. We can generate them directly from `weak_entity_spans`:

```python
import json
df = pd.read_csv("vaers_final_pipeline.csv")

label_studio_data = []

for _, row in df.iterrows():
    text = str(row.get("SYMPTOM_TEXT",""))
    spans = row.get("weak_entity_spans", "[]")
    try:
        spans = eval(spans) if isinstance(spans, str) else spans
    except:
        spans = []

    entities = []
    for ent in spans:
        entities.append({
            "start": ent["start"],
            "end": ent["end"],
            "label": [ent["label"]]
        })

    label_studio_data.append({
        "text": text,
        "entities": entities,
        "VAERS_ID": row["VAERS_ID"],
        "weak_label": row["weak_label"],
        "weak_label_probs": {
            "SEVERE": row["weak_label_prob_SEVERE"],
            "MODERATE": row["weak_label_prob_MODERATE"],
            "MILD": row["weak_label_prob_MILD"]
        }
    })

# Save JSONL for Label Studio
with open("vaers_label_studio.jsonl", "w") as f:
    for item in label_studio_data:
        f.write(json.dumps(item) + "\n")

print("✅ Saved JSONL for Label Studio review: vaers_label_studio.jsonl")
```

---

### 🔹 2️⃣ Import into Label Studio

1. Create a new **Labeling Project** in Label Studio.
2. Choose **NER labeling type**.
3. Upload `vaers_label_studio.jsonl` as **tasks**.
4. Configure **labels**: `ADE` and `DRUG`.
5. You can also display **row-level weak label** in the interface if you want to review the severity at the same time.

> Once reviewed, Label Studio allows you to export corrected spans → JSON/CoNLL.

---

### 🔹 3️⃣ Convert to CoNLL for BioBERT Multi-task

After reviewing in Label Studio, you can:

* Convert the annotated spans back into **BIO token sequences** (just like `text_to_bio` function)
* Keep your **severity label + probability** in parallel for **multi-task training**

Example CoNLL format per line:

```
Pain   B-ADE
gone   O
after  O
Fluzone B-DRUG
.      O
```

And row-level severity can be kept in a separate CSV mapping `VAERS_ID → severity` or included in a multi-task dataset loader.

---

✅ **Summary**

* `vaers_label_studio.jsonl` → for review and correction in Label Studio
* After Label Studio: export corrected spans → convert to BIO/CoNLL → use with your **multi-task BioBERT**:

  * **Token-level NER** (ADE/DRUG)
  * **Row-level severity classification/regression**

---

If you want, I can **write a full snippet to convert the Label Studio JSON export back into a multi-task dataset** ready for HuggingFace `datasets` for BioBERT **NER + severity**.

Do you want me to do that next?

Perfect ✅ — here’s a full snippet to **take Label Studio JSON export** and convert it into a **multi-task dataset** ready for HuggingFace **BioBERT NER + severity regression/classification**.

This assumes your Label Studio export looks like:

```json
{
  "text": "Pain gone after Fluzone.",
  "entities": [
    {"start": 0, "end": 4, "label": ["ADE"]},
    {"start": 15, "end": 22, "label": ["DRUG"]}
  ],
  "VAERS_ID": 810053,
  "weak_label": "Severe",
  "weak_label_probs": {"SEVERE":0.85,"MODERATE":0.1,"MILD":0.05}
}
```

---

### 🔹 1️⃣ Load Label Studio Export and Convert to BIO

```python
import json
import re
from sklearn.model_selection import train_test_split

# Severity mapping for classification/regression
severity2id = {"Severe": 0, "Moderate": 1, "Mild": 2}

# Load Label Studio JSON export
with open("vaers_label_studio_export.jsonl", "r") as f:
    tasks = [json.loads(line) for line in f]

def spans_to_bio(text, entities):
    """
    Convert Label Studio entity spans to BIO token labels
    """
    tokens = re.findall(r"\w+|[^\w\s]", text)
    labels = ["O"] * len(tokens)

    for ent in entities:
        ent_label = ent["label"][0]  # Label Studio uses list
        ent_tokens = re.findall(r"\w+|[^\w\s]", text[ent["start"]:ent["end"]])

        for i in range(len(tokens)):
            if tokens[i:i+len(ent_tokens)] == ent_tokens:
                labels[i] = f"B-{ent_label}"
                for j in range(1, len(ent_tokens)):
                    labels[i+j] = f"I-{ent_label}"
                break
    return tokens, labels

# Prepare dataset
multi_task_data = []
for task in tasks:
    text = task["text"]
    entities = task.get("entities", [])
    tokens, labels = spans_to_bio(text, entities)

    # Severity info (hard + soft)
    weak_label = task.get("weak_label", "Severe")
    weak_label_id = severity2id.get(weak_label, 0)
    weak_label_probs = task.get("weak_label_probs", {"SEVERE":1.0,"MODERATE":0.0,"MILD":0.0})

    multi_task_data.append({
        "id": task.get("VAERS_ID"),
        "tokens": tokens,
        "labels": labels,
        "severity_id": weak_label_id,
        "severity_probs": weak_label_probs
    })

# -----------------------------
# 2️⃣ Train/Val/Test Split
# -----------------------------
train, temp = train_test_split(multi_task_data, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save JSONL
def save_jsonl(filename, data):
    with open(filename,"w") as f:
        for item in data:
            f.write(json.dumps(item)+"\n")

save_jsonl("train_multi_task.jsonl", train)
save_jsonl("val_multi_task.jsonl", val)
save_jsonl("test_multi_task.jsonl", test)

print("✅ Multi-task dataset ready for BioBERT training (NER + severity)")
```

---

### 🔹 2️⃣ What You Get

Each row in the JSONL has:

```json
{
  "id": 810053,
  "tokens": ["Pain", "gone", "after", "Fluzone", "."],
  "labels": ["B-ADE", "O", "O", "B-DRUG", "O"],
  "severity_id": 0,
  "severity_probs": {"SEVERE":0.85,"MODERATE":0.1,"MILD":0.05}
}
```

* **`tokens` + `labels`** → token-level NER for BioBERT
* **`severity_id`** → hard classification label
* **`severity_probs`** → optional soft targets for regression / soft-label training

---

### 🔹 3️⃣ HuggingFace Dataset Integration Example

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "train_multi_task.jsonl",
    "validation": "val_multi_task.jsonl",
    "test": "test_multi_task.jsonl"
}, field="")

print(dataset)
```

Now you can use this **multi-task dataset** for BioBERT:

* **NER loss** → token-level cross-entropy
* **Severity loss** → row-level cross-entropy or MSE (if using probabilities as soft targets)

---

If you want, I can **write a full PyTorch/HuggingFace training loop** showing **joint NER + severity soft-label loss** using this dataset so you can start fine-tuning BioBERT directly.

Do you want me to do that next?
