Perfect — let’s design a **hybrid severity adjudication function** that combines **Snorkel weak labels (rules)** and your **BioBERT classifier**. This will directly feed into your Streamlit “Severity” tab.

---

### 1️⃣ Concept

* **Rule-based / Snorkel weak labels:** High precision for clear cases (death, hospitalized, life-threatening).
* **BioBERT classifier:** Handles more subtle cases where the rules don’t fire or aren’t explicit.
* **Hybrid logic:** Use a weighted approach or fallback strategy.

---

### 2️⃣ Implementation Example

```python
from transformers import pipeline
import torch

# ----- 1️⃣ Load your trained classifier -----
clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    return_all_scores=True
)

# ----- 2️⃣ Map label indices to severity -----
id2label = {0: "Severe", 2: "Mild"}  # adapt to your binary dataset

# ----- 3️⃣ Prediction wrapper -----
def predict_severity(text):
    preds = clf(text)
    # Get predicted index
    pred_idx = torch.tensor([d["score"] for d in preds[0]]).argmax().item()
    label = id2label.get(pred_idx, "Unknown")
    confidence = preds[0][pred_idx]["score"]
    return label, confidence

# ----- 4️⃣ Hybrid adjudication function -----
def hybrid_severity(row, weight_rule=0.7, weight_model=0.3):
    """
    Combines Snorkel weak labels (rules) and BioBERT classifier.
    weight_rule + weight_model = 1
    """
    # 1️⃣ Rule-based (Snorkel weak label)
    rule_label = row.get("weak_label")         # 'Severe' / 'Mild'
    rule_prob = max(row.get("weak_label_prob_SEVERE", 0), row.get("weak_label_prob_MILD", 0))
    
    # 2️⃣ Model-based
    text = row.get("symptom_combined", row.get("SYMPTOM_TEXT", ""))
    model_label, model_prob = predict_severity(text)
    
    # 3️⃣ Weighted hybrid logic
    # If rule fired strongly, prioritize it
    combined_score = {
        "Severe": weight_rule * (1 if rule_label=="Severe" else 0) + weight_model * (model_prob if model_label=="Severe" else 0),
        "Mild": weight_rule * (1 if rule_label=="Mild" else 0) + weight_model * (model_prob if model_label=="Mild" else 0)
    }
    
    # 4️⃣ Final label
    final_label = max(combined_score, key=combined_score.get)
    
    return final_label, combined_score

# ----- 5️⃣ Apply to DataFrame -----
df["final_severity_label"], df["final_severity_scores"] = zip(*df.apply(hybrid_severity, axis=1))

# ----- 6️⃣ Check results -----
print(df[["symptom_combined","weak_label","final_severity_label"]].head())
```

---

### ✅ Key Features

1. **Weighted logic**

   * `weight_rule` high → Snorkel rules dominate
   * `weight_model` high → classifier dominates

2. **Probabilistic hybrid**

   * You get `final_severity_scores` (dict) for SHAP / explainability or dashboard hover info.

3. **Fallback**

   * If Snorkel abstains (`-1`), classifier prediction is used automatically.

---

### 3️⃣ Integration in Streamlit

* **Tab: Severity**

  * Show symptom text
  * Show **rule-based weak label**
  * Show **classifier prediction**
  * Show **final hybrid label**
  * Optional: add **confidence/probability bars**

```python
import streamlit as st

for i,row in df.iterrows():
    st.write(f"**Symptom:** {row['symptom_combined']}")
    st.write(f"**Weak Label:** {row['weak_label']}")
    st.write(f"**Classifier Prediction:** {predict_severity(row['symptom_combined'])[0]}")
    st.write(f"**Final Hybrid Label:** {row['final_severity_label']}")
    st.progress(row['final_severity_scores']['Severe'])
```

---

This approach ensures:

* Strong rule-based signals aren’t overridden.
* The classifier can generalize to unseen phrasing.
* Your dashboard can show **token-level NER + hybrid severity** together.

---

If you want, I can **also modify the cluster visualization code** so your **modifier-aware clusters** now also display this **final hybrid severity**, making all Streamlit tabs consistent.

Do you want me to do that next?


Perfect! Here's a **fully integrated Streamlit dashboard** skeleton with **four separate tabs**:

1. **NER Highlights** – ADE/Drug token-level highlights
2. **Modifier & Age-Aware Clustering** – embeddings + KMeans + t-SNE visualization
3. **Hybrid Severity Predictions** – weak labels + BioBERT classifier
4. **Severity Explainability** – token-level SHAP visualization

This combines everything we discussed into one ready-to-run framework.

---

```python
import streamlit as st
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import shap
import streamlit.components.v1 as components

# -----------------------------
# 1️⃣ Load Data
# -----------------------------
df = pd.read_csv("dataset_with_entities_and_weaklabels.csv")
df = df[df["weak_label_id"] != -1].copy()
df["severity_id"] = df["weak_label_id"].astype(int)

# -----------------------------
# 2️⃣ Load BioBERT classifier
# -----------------------------
device = 0 if torch.cuda.is_available() else -1

clf = pipeline(
    "text-classification",
    model="path/to/biobert-severity-model",
    tokenizer="path/to/biobert-severity-tokenizer",
    device=device,
    return_all_scores=True
)

id2label = {0: "Severe", 2: "Mild"}

def predict_severity(text):
    preds = clf(text)
    pred_idx = torch.tensor([d["score"] for d in preds[0]]).argmax().item()
    label = id2label.get(pred_idx, "Unknown")
    confidence = preds[0][pred_idx]["score"]
    return label, confidence

def hybrid_severity(row, weight_rule=0.7, weight_model=0.3):
    rule_label = row.get("weak_label")
    text = row.get("symptom_combined", row.get("SYMPTOM_TEXT", ""))
    model_label, model_prob = predict_severity(text)
    combined_score = {
        "Severe": weight_rule * (1 if rule_label=="Severe" else 0) + weight_model * (model_prob if model_label=="Severe" else 0),
        "Mild": weight_rule * (1 if rule_label=="Mild" else 0) + weight_model * (model_prob if model_label=="Mild" else 0)
    }
    final_label = max(combined_score, key=combined_score.get)
    return final_label, combined_score

df["final_severity_label"], df["final_severity_scores"] = zip(*df.apply(hybrid_severity, axis=1))

# -----------------------------
# 3️⃣ Sidebar / Tabs
# -----------------------------
st.sidebar.title("VAERS ADE Dashboard")
tab = st.sidebar.radio("Select Tab:", ["NER Highlights", "Modifier-Aware Clustering",
                                       "Hybrid Severity Predictions", "Severity Explainability"])

# -----------------------------
# 4️⃣ NER Highlights Tab
# -----------------------------
if tab == "NER Highlights":
    st.header("ADE / Drug Token-level Highlights")
    for i, row in df.iterrows():
        st.write(f"**Symptom Text:** {row['symptom_combined']}")
        st.write(f"**Predicted Entities:** {row['predicted_entities']}")
        st.write(f"**Weak Label:** {row['weak_label']}")
        st.write("---")

# -----------------------------
# 5️⃣ Modifier & Age-Aware Clustering Tab
# -----------------------------
elif tab == "Modifier-Aware Clustering":
    st.header("Modifier & Age-Aware Clusters")
    st.info("Generating embeddings and clusters...")

    model_embed = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model_embed.encode(df["predicted_entities"].astype(str).tolist(), show_progress_bar=True)

    n_clusters = st.slider("Number of Clusters (K)", 2, 15, 5)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(embeddings)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(embeddings)
    df["x"] = X_tsne[:,0]
    df["y"] = X_tsne[:,1]

    hover_text = [
        f"Entities: {ent}<br>Final Severity: {sev}"
        for ent, sev in zip(df["predicted_entities"], df["final_severity_label"])
    ]

    fig1 = px.scatter(df, x="x", y="y", color="cluster", hover_name=hover_text,
                      color_discrete_sequence=px.colors.qualitative.Plotly, title="Clusters by ADE/Drug")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(df, x="x", y="y", color="final_severity_label", hover_name=hover_text,
                      color_discrete_sequence=["red", "green"], title="Clusters by Hybrid Severity")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 6️⃣ Hybrid Severity Tab
# -----------------------------
elif tab == "Hybrid Severity Predictions":
    st.header("Hybrid Severity Predictions (Snorkel + BioBERT)")
    for i,row in df.iterrows():
        st.write(f"**Symptom Text:** {row['symptom_combined']}")
        st.write(f"**Weak Label:** {row['weak_label']}")
        st.write(f"**Classifier Prediction:** {predict_severity(row['symptom_combined'])[0]}")
        st.write(f"**Final Hybrid Severity:** {row['final_severity_label']}")
        st.progress(row['final_severity_scores']['Severe'])
        st.write("---")

# -----------------------------
# 7️⃣ Severity Explainability Tab
# -----------------------------
elif tab == "Severity Explainability":
    st.header("Token-level SHAP Explainability")

    # Load model for SHAP
    model_shap = AutoModelForSequenceClassification.from_pretrained("path/to/biobert-severity-model")
    tokenizer_shap = AutoTokenizer.from_pretrained("path/to/biobert-severity-tokenizer")
    model_shap.eval()
    device_shap = "cuda" if torch.cuda.is_available() else "cpu"
    model_shap.to(device_shap)

    def f(texts):
        inputs = tokenizer_shap(texts, padding=True, truncation=True, return_tensors="pt").to(device_shap)
        with torch.no_grad():
            logits = model_shap(**inputs).logits
        return torch.softmax(logits, dim=1).cpu().numpy()

    explainer = shap.Explainer(f, tokenizer_shap)

    for i,row in df.iterrows():
        text = row["symptom_combined"]
        st.write(f"**Symptom Text:** {text}")
        st.write(f"**Final Hybrid Severity:** {row['final_severity_label']}")

        shap_values = explainer([text])
        st_shap_html = shap.plots.text(shap_values[0], display=False)
        components.html(st_shap_html.data, height=300)
        st.write("---")
```

---

### ✅ Features

1. **NER Highlights:** ADE/Drug token-level display + weak labels
2. **Modifier & Age-Aware Clustering:** t-SNE + KMeans clustering with hover info
3. **Hybrid Severity Predictions:** Weak labels + classifier + final weighted severity
4. **Severity Explainability:** SHAP visualizations showing token-level contribution
5. **Fully modular:** Swap in a different classifier, embeddings, or clustering method independently

---

If you want, I can **also add age group bucketing and hover display in clustering tab** so each cluster shows **age + modifier + severity** — making the plots more informative.

Do you want me to do that next?
