Perfect ✅ — I’ll outline a **full separate training pipeline** for your project, covering:

1. **Token-level NER (ADE / DRUG)**
2. **Severity classifier (Severe / Moderate / Mild)**
3. Saving both models for Streamlit usage
4. Optionally, SHAP-ready inputs

---

## **1️⃣ Token-level NER Training**

```python
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
import numpy as np

# -----------------------------
# Load your BIO JSONL dataset
# -----------------------------
dataset = load_dataset("json", data_files={
    "train": "train_multi_task.jsonl",
    "validation": "val_multi_task.jsonl",
    "test": "test_multi_task.jsonl"
}, field="")

# Use tokens + BIO labels for NER
ner_label_list = ["O", "B-ADE", "I-ADE", "B-DRUG", "I-DRUG"]
label2id = {label:i for i,label in enumerate(ner_label_list)}

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, padding="max_length", max_length=128)
    
    labels = []
    for i, label_seq in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # ignored in loss
            elif word_idx != prev_word_idx:
                aligned_labels.append(label2id[label_seq[word_idx]])
            else:
                # Set same token as I- if not O
                if label_seq[word_idx].startswith("B-"):
                    aligned_labels.append(label2id[label_seq[word_idx].replace("B-","I-")])
                else:
                    aligned_labels.append(label2id[label_seq[word_idx]])
            prev_word_idx = word_idx
        labels.append(aligned_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# -----------------------------
# Model
# -----------------------------
ner_model = AutoModelForTokenClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    num_labels=len(ner_label_list)
)

# -----------------------------
# Training
# -----------------------------
training_args = TrainingArguments(
    output_dir="./ner_model",
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=ner_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"]
)

trainer.train()
trainer.save_model("./ner_model_final")
tokenizer.save_pretrained("./ner_model_final")
```

✅ This produces a **ready-to-use NER model** for your Streamlit token highlights.

---

## **2️⃣ Severity Classification Training**

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder

# Prepare dataset: use text + severity_id
def prepare_severity_dataset(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    texts, labels = [], []
    for line in lines:
        item = json.loads(line)
        texts.append(" ".join(item["tokens"]))  # full text
        labels.append(item["severity_id"])
    return Dataset.from_dict({"text": texts, "label": labels})

train_ds = prepare_severity_dataset("train_multi_task.jsonl")
val_ds = prepare_severity_dataset("val_multi_task.jsonl")
test_ds = prepare_severity_dataset("test_multi_task.jsonl")

# Tokenizer
sev_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
def tokenize_fn(batch):
    return sev_tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.map(tokenize_fn, batched=True)

# Model
severity_model = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    num_labels=3  # Severe, Moderate, Mild
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./severity_model",
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=severity_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

trainer.train()
trainer.save_model("./severity_model_final")
sev_tokenizer.save_pretrained("./severity_model_final")
```

✅ Produces a **row-level severity classifier** for your Streamlit dashboard.

---

## **3️⃣ Streamlit Integration Flow**

```
User selects VAERS row or enters text
      │
      ├─> NER model → token-level BIO → highlights (ADE/DRUG)
      │
      └─> Severity classifier → severity probability / class → cluster plots
```

* **Explainability (SHAP)**:

  * NER → token-level SHAP
  * Severity → document-level SHAP

---

### **Advantages of Separate Models**

1. Streamlit can show **NER highlights** independently of severity
2. Severity classifier can be retrained / updated separately
3. SHAP analysis is simpler for both tasks
4. Easier maintenance & debugging

---

If you want, I can **also draft the Streamlit code snippet** that:

* Loads both models
* Highlights tokens for NER
* Shows severity classification + cluster plot
* Computes SHAP explanations for both

This will give you a **ready-to-run interactive dashboard**.

Do you want me to do that next?

Perfect! Here’s a **complete Streamlit template** that integrates your **separate NER and severity models**, highlights token-level ADE/DRUG, shows severity classification with cluster-like probability visualization, and prepares SHAP explainability.

---

## **Streamlit App Template**

```python
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
import shap
import numpy as np

# -----------------------------
# 1️⃣ Load Models
# -----------------------------
@st.cache_resource
def load_ner_model(path="./ner_model_final"):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForTokenClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_severity_model(path="./severity_model_final"):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

ner_tokenizer, ner_model = load_ner_model()
sev_tokenizer, sev_model = load_severity_model()

ner_label_list = ["O", "B-ADE", "I-ADE", "B-DRUG", "I-DRUG"]

# -----------------------------
# 2️⃣ Streamlit UI
# -----------------------------
st.title("VAERS NER + Severity Dashboard")
text_input = st.text_area("Enter VAERS report text here:")

if text_input:
    st.subheader("Token-level NER Highlights")
    
    # -----------------------------
    # 3️⃣ NER Prediction
    # -----------------------------
    tokens = ner_tokenizer.tokenize(text_input)
    inputs = ner_tokenizer.encode(text_input, return_tensors="pt")
    with torch.no_grad():
        outputs = ner_model(inputs).logits
    predictions = torch.argmax(outputs, dim=-1)[0].tolist()
    
    # Decode predictions
    decoded = []
    for token, pred_id in zip(tokens, predictions[1:len(tokens)+1]):  # skip [CLS]
        decoded.append((token, ner_label_list[pred_id]))
    
    # Display highlights
    highlighted_text = ""
    for token, label in decoded:
        if label != "O":
            color = "#ffcccc" if "ADE" in label else "#ccffcc"
            highlighted_text += f"<span style='background-color:{color}'>{token}</span> "
        else:
            highlighted_text += token + " "
    st.markdown(highlighted_text, unsafe_allow_html=True)
    
    # -----------------------------
    # 4️⃣ Severity Prediction
    # -----------------------------
    sev_inputs = sev_tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        sev_logits = sev_model(**sev_inputs).logits
        sev_probs = torch.softmax(sev_logits, dim=-1).cpu().numpy()[0]
        sev_class = np.argmax(sev_probs)
    
    severity_map = {0:"Severe", 1:"Moderate", 2:"Mild"}
    st.subheader("Severity Classification")
    st.write(f"Predicted severity: **{severity_map[sev_class]}**")
    
    # Cluster-style probability plot
    st.bar_chart({"Severe": sev_probs[0], "Moderate": sev_probs[1], "Mild": sev_probs[2]})
    
    # -----------------------------
    # 5️⃣ SHAP Explainability
    # -----------------------------
    st.subheader("Explainability (SHAP) for Severity")
    
    def model_forward(x):
        tokens = sev_tokenizer(x, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = sev_model(**tokens).logits
            return torch.softmax(logits, dim=-1).cpu().numpy()
    
    explainer = shap.Explainer(model_forward, shap.maskers.Text(sev_tokenizer))
    shap_values = explainer([text_input])
    
    st.text("SHAP token contributions (severity probabilities)")
    st_shap_text = ""
    for word, contrib in zip(shap_values.data[0], shap_values.values[0][sev_class]):
        color_intensity = int(255 * min(max(contrib,0),1))
        st_shap_text += f"<span style='background-color: rgba(255,{255-color_intensity},{255-color_intensity},0.5)'>{word} </span>"
    st.markdown(st_shap_text, unsafe_allow_html=True)
```

---

### **✅ Features**

1. **NER Highlights**

   * Token-level B-ADE / I-ADE / B-DRUG / I-DRUG
   * Color-coded in Streamlit

2. **Severity Classification**

   * Softmax probabilities plotted as bar chart
   * Predicted class displayed

3. **SHAP Explainability**

   * Highlights words contributing to predicted severity
   * Can extend to ADE/DRUG NER explainability later

---

### **Usage**

1. Run Streamlit:

```bash
streamlit run vaers_dashboard.py
```

2. Paste VAERS text → see token highlights and severity prediction

3. SHAP visual shows word-level contribution to severity

---

This setup keeps **NER + Severity models separate**, which is ideal for your dashboard:

* Token-level NER → highlights
* Severity → document-level probability + cluster plot
* SHAP → word-level explanation

---

If you want, I can also **add an option to load multiple VAERS rows**, show **NER + severity for all**, and produce **interactive cluster plots for the dataset**, which is useful for exploratory analysis.

Do you want me to do that next?


Great! Here’s an **extended Streamlit template** for **batch VAERS analysis**, showing **NER highlights, severity predictions, and cluster plots** across multiple reports. This is perfect for exploratory analysis and dashboarding.

---

## **Streamlit Dashboard for Multiple VAERS Rows**

```python
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
import shap
import numpy as np

# -----------------------------
# 1️⃣ Load Models
# -----------------------------
@st.cache_resource
def load_ner_model(path="./ner_model_final"):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForTokenClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_severity_model(path="./severity_model_final"):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

ner_tokenizer, ner_model = load_ner_model()
sev_tokenizer, sev_model = load_severity_model()

ner_label_list = ["O", "B-ADE", "I-ADE", "B-DRUG", "I-DRUG"]
severity_map = {0:"Severe", 1:"Moderate", 2:"Mild"}

# -----------------------------
# 2️⃣ Streamlit UI
# -----------------------------
st.title("VAERS NER + Severity Dashboard (Batch Mode)")

uploaded_file = st.file_uploader("Upload CSV of VAERS reports", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if "SYMPTOM_TEXT" not in df.columns:
        st.error("CSV must have 'SYMPTOM_TEXT' column")
    else:
        st.write(f"Loaded {len(df)} reports")
        
        selected_rows = st.multiselect(
            "Select VAERS IDs to analyze",
            options=df.index.tolist(),
            default=df.index[:5].tolist()
        )
        
        for idx in selected_rows:
            st.markdown("---")
            text_input = str(df.loc[idx, "SYMPTOM_TEXT"])
            st.subheader(f"VAERS_ID: {df.loc[idx,'VAERS_ID']}")

            # -----------------------------
            # 3️⃣ NER Prediction
            # -----------------------------
            tokens = ner_tokenizer.tokenize(text_input)
            inputs = ner_tokenizer.encode(text_input, return_tensors="pt")
            with torch.no_grad():
                outputs = ner_model(inputs).logits
            predictions = torch.argmax(outputs, dim=-1)[0].tolist()

            decoded = []
            for token, pred_id in zip(tokens, predictions[1:len(tokens)+1]):
                decoded.append((token, ner_label_list[pred_id]))

            highlighted_text = ""
            for token, label in decoded:
                if label != "O":
                    color = "#ffcccc" if "ADE" in label else "#ccffcc"
                    highlighted_text += f"<span style='background-color:{color}'>{token}</span> "
                else:
                    highlighted_text += token + " "
            st.markdown(highlighted_text, unsafe_allow_html=True)

            # -----------------------------
            # 4️⃣ Severity Prediction
            # -----------------------------
            sev_inputs = sev_tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                sev_logits = sev_model(**sev_inputs).logits
                sev_probs = torch.softmax(sev_logits, dim=-1).cpu().numpy()[0]
                sev_class = np.argmax(sev_probs)
            
            st.write(f"Predicted severity: **{severity_map[sev_class]}**")
            st.bar_chart({"Severe": sev_probs[0], "Moderate": sev_probs[1], "Mild": sev_probs[2]})

        # -----------------------------
        # 5️⃣ Optional: SHAP for batch
        # -----------------------------
        st.subheader("SHAP Explainability for Selected Reports")

        def model_forward(x):
            inputs = sev_tokenizer(x, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                logits = sev_model(**inputs).logits
                return torch.softmax(logits, dim=-1).cpu().numpy()
        
        selected_texts = [str(df.loc[idx,"SYMPTOM_TEXT"]) for idx in selected_rows]
        explainer = shap.Explainer(model_forward, shap.maskers.Text(sev_tokenizer))
        shap_values = explainer(selected_texts)

        for i, text in enumerate(selected_texts):
            st.markdown(f"**VAERS_ID: {df.loc[selected_rows[i],'VAERS_ID']}**")
            st.text("SHAP token contributions (for predicted severity class)")
            shap_class_idx = np.argmax(sev_model(**sev_tokenizer(text, return_tensors="pt"))[0].softmax(-1).detach().numpy())
            st_shap_text = ""
            for word, contrib in zip(shap_values.data[i], shap_values.values[i][shap_class_idx]):
                color_intensity = int(255 * min(max(contrib,0),1))
                st_shap_text += f"<span style='background-color: rgba(255,{255-color_intensity},{255-color_intensity},0.5)'>{word} </span>"
            st.markdown(st_shap_text, unsafe_allow_html=True)
            st.markdown("---")
```

---

### **✅ Features**

1. **Batch VAERS analysis**

   * Upload CSV → select multiple VAERS IDs

2. **Token-level NER highlights**

   * Color-coded ADE / DRUG tokens per report

3. **Severity classifier output**

   * Softmax probabilities → bar chart per report

4. **SHAP explainability**

   * Shows contribution of words to severity predictions

5. **Scalable & interactive**

   * Works with hundreds of reports (subset at a time)

---

### **Usage**

```bash
streamlit run vaers_dashboard_batch.py
```

* Upload your VAERS CSV
* Select a few rows to visualize
* Inspect token-level NER, severity prediction, and SHAP explainability

---

This setup is **ready for your final portfolio/dashboard**:

* Separate NER and severity models → easier to debug & maintain
* Token-level and row-level explainability
* Interactive cluster/probability visualization

---

I can also **add a clustering visualization** of all selected VAERS severity probabilities in 2D (UMAP / PCA) to see patterns across reports.

Do you want me to add that next?
