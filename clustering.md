Perfect! Let’s design a **Streamlit-ready pipeline** that achieves:

* **NER tab** → show ADE/Drug spans
* **Clustering tab** → age-aware & modifier-aware clusters
* **Severity tab** → weak-label-based severity with SHAP explanations

…but **without copying the heavy semantic embedding + lexicon approach** you showed.

---

## **1️⃣ Prepare Data**

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import streamlit as st
import plotly.express as px

# Load weak-labeled dataset
df = pd.read_csv("dataset_with_entities_and_weaklabels.csv")

# Ensure predicted_entities column exists (list of ADEs per row)
df['predicted_entities'] = df['ADE'].apply(lambda x: eval(x) if isinstance(x,str) else [])

# Age bucket
def classify_age_group(age):
    if pd.isna(age):
        return 'Unknown'
    elif age < 18:
        return 'Child'
    elif age < 60:
        return 'Adult'
    return 'Senior'

df['age_group'] = df['AGE_YRS'].apply(classify_age_group)
```

---

## **2️⃣ Optional Modifier Extraction (lightweight)**

```python
# Simple regex-based modifier detection around ADE spans
modifier_map = {'high': ['severe','critical','life-threatening'],
                'medium': ['moderate','persistent','significant'],
                'low': ['mild','slight','minor']}

def detect_modifier(text, ade):
    if not ade: return 'medium'  # default
    ade_lower = [a.lower() for a in ade]
    text_lower = text.lower()
    for level, words in modifier_map.items():
        for w in words:
            if w in text_lower:
                return level
    return 'medium'

df['modifier'] = df.apply(lambda r: detect_modifier(r['SYMPTOM_TEXT'], r['predicted_entities']), axis=1)

modifier_num = {'low':0,'medium':1,'high':2}
df['modifier_num'] = df['modifier'].map(modifier_num)
```

---

## **3️⃣ Embed ADEs for Clustering**

```python
model_embed = SentenceTransformer("all-MiniLM-L6-v2")

# Flatten ADEs into one string per row
df['ade_text'] = df['predicted_entities'].apply(lambda x: " ".join(x) if x else "")

# Generate embeddings
embeddings = model_embed.encode(df['ade_text'].tolist(), normalize_embeddings=True)
```

---

## **4️⃣ Combine features for clustering**

```python
# Age as numeric feature
age_num = df['AGE_YRS'].fillna(35).to_numpy().reshape(-1,1)/100.0

# Modifier as numeric feature
modifier_feat = df['modifier_num'].to_numpy().reshape(-1,1)

# Feature matrix = embeddings + age + modifier
X = np.hstack([embeddings, age_num, modifier_feat])

# KMeans clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)
```

---

## **5️⃣ t-SNE for Visualization**

```python
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(embeddings)
df['x'] = X_tsne[:,0]
df['y'] = X_tsne[:,1]

hover_text = [
    f"ADEs: {ade}<br>Age: {age}<br>Modifier: {mod}<br>Severity: {sev}"
    for ade, age, mod, sev in zip(df['predicted_entities'], df['age_group'], df['modifier'], df['weak_label'])
]
```

---

## **6️⃣ Streamlit Cluster Plots**

```python
st.subheader("Clusters by KMeans")
fig1 = px.scatter(df, x='x', y='y', color='cluster',
                  hover_name=hover_text,
                  title="ADE Clusters")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Clusters by Severity")
fig2 = px.scatter(df, x='x', y='y', color='weak_label',
                  hover_name=hover_text,
                  color_discrete_sequence=px.colors.qualitative.Set1,
                  title="Clusters colored by Weak Label Severity")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Clusters by Age Group")
fig3 = px.scatter(df, x='x', y='y', color='age_group',
                  hover_name=hover_text,
                  color_discrete_sequence=px.colors.qualitative.Set2,
                  title="Clusters colored by Age Group")
st.plotly_chart(fig3, use_container_width=True)
```

---

### ✅ **Advantages of this approach**

1. **Lightweight modifier handling** → regex + weak defaults; no semantic embeddings needed.
2. **Age-aware clusters** → age numeric or bucketed feature.
3. **Weak-label severity** → used for coloring and classifier training.
4. **Simple embedding-based ADE clustering** → semantic relationships preserved.
5. **Streamlit-ready** → tabs can be:

   * NER highlights (`predicted_entities`)
   * Clustering (above plots)
   * Severity classifier + SHAP explanations (from weak labels)

---

If you want, I can **also sketch a Streamlit layout with 4 clean tabs**, showing **NER, Clustering, Severity prediction, and SHAP explanations**, integrating this clustering approach.

Do you want me to do that?
Exactly ✅ — that’s the clean separation:

---

### **1️⃣ Clustering (modifier-aware + age-aware)**

* Uses **NER-extracted ADEs**.
* Uses **lightweight modifier detection** (regex/keywords or even “medium” default if missing).
* Uses **age buckets / numeric age**.
* Embeddings of ADE spans → semantic similarity clustering.
* **Purpose:** Visualization and grouping of ADEs by similarity, age, and modifier hints.
* **Does not rely on weak labels** for training — only for coloring/hover info if you like.

---

### **2️⃣ Severity Classifier (Snorkel + BioBERT)**

* Use your **Snorkel weak labels** (`weak_label_id`) as **supervised signal**.
* Inputs: `SYMPTOM_TEXT` (and optionally predicted entities).
* Labels: `weak_label_id` → 0=Severe, 1=Moderate, 2=Mild.
* Fine-tune a **BioBERT / domain-specific model** to predict severity.
* SHAP or LIME explainability → why the model predicted Severe/Moderate/Mild.

**Key idea:**

* **Clustering** = unsupervised grouping, helpful for visualization & exploratory analysis.
* **Classifier** = supervised model trained on weak labels, can generalize to new texts, and can produce SHAP explanations for downstream UI.

---

So yes: your pipeline will look like:

| Tab                 | Data / Input               | Method                                                          |
| ------------------- | -------------------------- | --------------------------------------------------------------- |
| NER                 | ADE/Drug spans             | BioBERT or NER model                                            |
| Clustering          | ADE spans + modifier + age | Embeddings + KMeans/HDBSCAN + coloring by modifier/age/severity |
| Severity Classifier | Symptom text               | BioBERT fine-tuned on Snorkel weak labels                       |
| SHAP/Explainability | Classifier predictions     | SHAP / LIME explanations for severity predictions               |

---

If you want, I can **draw a small Mermaid.js diagram** showing **flow from NER → clustering → weak labels → classifier → SHAP**. This often helps clarify the separation visually.

Do you want me to do that?
