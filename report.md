## 🧩 **ADEGuard: Hybrid ADE Detection & Severity Analysis**

### 🧠 Overview

This Streamlit dashboard performs three major tasks:

1. **ADE/Drug Named Entity Recognition (NER)** using BioBERT
2. **Severity Classification** (Severe / Moderate / Mild) using a Transformer model
3. **Explainability with SHAP** and **clustering of ADE patterns** using Sentence Embeddings

The goal is to detect and interpret Adverse Drug Events (ADEs) automatically from clinical text and explain model behavior.

---

## 🔧 1️⃣ Streamlit Setup and Configuration

```python
st.set_page_config(layout="wide", page_title="ADEGuard Dashboard")
st.title("🧠 ADEGuard 🧠")
st.subheader("Hybrid ADE Detection & Severity Analysis")
```

* Sets up the Streamlit page layout and title.
* The “wide” layout gives more space for charts and text outputs.
* The title and subheader define the dashboard branding.

---

## 📤 2️⃣ Sidebar — Uploading Input Data

```python
uploaded_file = st.sidebar.file_uploader("Upload CSV with columns 'symptom_text' and 'AGE'", type=["csv"])
```

* The user uploads a CSV file containing **clinical symptom text** and **patient age**.
* The uploaded dataset is read into a Pandas DataFrame.

```python
if uploaded_file is None:
    st.info("👈 Please upload a CSV file to start analysis.")
    st.stop()
```

* Prevents the rest of the dashboard from running until data is provided.

---

## 👩‍⚕️ 3️⃣ Age Grouping Logic

```python
def age_group(age):
    ...
df["age_group"] = df["age"].apply(age_group)
```

* Converts raw numeric age into **categorical bins**:

  * `Child`, `Young Adult`, `Middle Age`, `Senior`
* This feature is later used in **clustering** and **interpretation** (e.g., to see ADE severity trends across age groups).

---

## 🧭 4️⃣ Tabs — Dashboard Navigation

```python
tabs = st.tabs(["NER", "Severity Prediction & Explainability", "Clustering"])
```

* Streamlit tabs separate functionalities neatly:

  * **Tab 1:** NER — extracts ADEs and Drug mentions.
  * **Tab 2:** Severity Classification + Explainability.
  * **Tab 3:** Clustering — pattern discovery and visualization.

---

## 🩸 5️⃣ Tab 1 — ADE/Drug Detection (NER)

### 🔹 Model Loading

```python
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
```

* Loads your **BioBERT-based NER model**.
* This model is fine-tuned to detect **ADE** (adverse event) and **DRUG** tokens.

---

### 🔹 Entity Extraction

```python
outputs = model(**encoded)
predictions = torch.argmax(outputs.logits, dim=-1)
```

* Performs token-level classification for each word.
* The model assigns labels like `B-ADE`, `I-ADE`, `B-DRUG`, `I-DRUG`, `O`.

Then tokens are grouped:

* `B/I` tokens are merged to reconstruct entity phrases (e.g., *“severe rash”* as one ADE).

---

### 🔹 Displaying NER Results

```python
st.dataframe(df[["symptom_text", "age_group", "ADE", "DRUG"]].head(10))
```

* Shows extracted ADEs and drugs for quick inspection.

### 🔹 Token-level Highlight

```python
html_text += f'<span style="background-color:{color}; ...>{escape(token)}</span>'
```

* Displays color-coded highlights:

  * **Red:** ADE tokens
  * **Blue:** Drug tokens

This helps clinicians **visually verify** the extraction quality.

---

## ⚖️ 6️⃣ Tab 2 — Severity Classification + Explainability

### 🔹 Model Loading

```python
tokenizer, model = load_classifier(C_MODEL_PATH)
```

* Loads the fine-tuned **sequence classification model** (e.g., DistilBERT or BioClinicalBERT).
* This model predicts **Mild / Moderate / Severe** from each symptom text.

---

### 🔹 Severity Prediction

```python
preds = clf(text)
scores = [d["score"] for d in preds[0]]
label = id2label.get(np.argmax(scores))
```

* Uses a Hugging Face pipeline for classification.
* Returns a severity label + confidence scores.

---

### 🔹 Display Predictions

```python
st.dataframe(df[["symptom_text", "pred_label"]])
```

* Displays each input text alongside its predicted severity.

---

## 🧩 7️⃣ SHAP Explainability

### 🔹 What is SHAP?

SHAP (SHapley Additive exPlanations) quantifies **how much each token contributes** to the model’s decision.

---

### 🔹 Explainer Creation

```python
explainer = shap.Explainer(clf)
shap_values = explainer([example_text])
```

* The `shap.Explainer` wraps your text-classification pipeline.
* Computes per-token importance values showing **positive** or **negative influence** toward severity.

---

### 🔹 Visualizing SHAP Outputs

```python
text_html = shap.plots.text(shap_values[0], display=False)
```

* Generates token-level visualization (heatmap-style HTML).
* Redder tokens = stronger contribution to “Severe”.

---

### 🔹 Word-level Aggregation

```python
word_scores.append(np.mean(np.abs(current_vals)))
```

* Aggregates subword tokens (e.g., `head`, `##ache`) into one word.
* Normalizes scores for consistent visualization.

---

### 🔹 Highlighting Important Tokens

```python
color = f"rgba(255,0,0,{0.3 + 0.7 * score})"
```

* Tokens are shaded red based on their SHAP importance.
* Helps explain **why** the model thinks a sentence is “Severe” or “Mild”.

---

### 🔹 Importance Chart

```python
st.bar_chart(importance_df.set_index("Word"))
```

* Shows a **bar chart** ranking tokens by importance.
* Makes interpretability easy for non-technical stakeholders (e.g., doctors, regulators).

---

## 🔍 8️⃣ (In Your Full Version) — Clustering (Hybrid ADE Pattern Analysis)

Although not included in this code snippet, your clustering tab:

* Uses **SentenceTransformer embeddings** (`all-MiniLM-L6-v2`) to convert entity text into vector space.
* Groups similar ADE/Drug cases using **K-Means + t-SNE**.
* Colors clusters by **severity, age group, or source** to find interpretable clinical patterns.

---

## 🧭 Summary — How It All Fits Together

| Module                    | Input                       | Model                                 | Output           | Purpose             |
| ------------------------- | --------------------------- | ------------------------------------- | ---------------- | ------------------- |
| **NER**                   | Symptom text                | BioBERT (Token Classification)        | ADEs & Drugs     | Identify entities   |
| **Severity Classifier**   | Symptom text                | Transformer (Sequence Classification) | Severity label   | Risk stratification |
| **Explainability (SHAP)** | Model predictions           | SHAP Explainer                        | Token importance | Transparency        |
| **Clustering**            | ADE + Drug + Severity + Age | Sentence Transformer + K-Means        | Clustered plots  | Pattern discovery   |

---

Would you like me to next write this as a **formal project report explanation (PDF-ready)** — with section headings, diagrams (e.g., pipeline diagram), and short paragraphs under each (suitable for submission or presentation)?
Perfect — this code block is already doing **most of what a hybrid explainable ADE (Adverse Drug Event) clustering pipeline** should do 👏

Let’s break it down so you know **exactly what’s happening** and how to **read the output plots**, because you’ve now combined **semantic clustering + severity classification + metadata explainability** in a single Streamlit tab.

---

## 🧭 1️⃣ What’s Actually Happening Step-by-Step

### Step 1: Create Sentence Embeddings

```python
model_embed = SentenceTransformer("all-MiniLM-L6-v2")
df["entities_text"] = df.apply(lambda r: " ".join(r["ADE"] + r["DRUG"]), axis=1)
embeddings = model_embed.encode(df["entities_text"].astype(str).tolist(), show_progress_bar=True)
```

* Each `ADE + DRUG` text is converted into a **768-dimensional embedding** by the Sentence Transformer.
* These embeddings capture the *meaning* of the text — so “fever and headache” and “nausea and chills” end up *close together* in this semantic space.

🔹 **Analogy:** It’s like mapping each patient report into a “meaning space” where similar side effects cluster together.

---

### Step 2: Assign Severity (Hybrid)

```python
results = df["symptom_text"].apply(hybrid_severity_explain)
df[["modifier", "severity_source"]] = pd.DataFrame(results.tolist(), index=df.index)
```

* Uses a **classifier (BioBERT model)** if available.
* If not, falls back to **rule-based heuristics** (e.g., if text contains “severe” or “hospitalized”).
* Assigns:

  * `modifier` = high / medium / low
  * `severity_source` = classifier or rule-based

✅ This gives each sample a severity level — either learned or inferred.

---

### Step 3: Cluster Semantically Similar Reports

```python
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df["cluster"] = kmeans.fit_predict(embeddings)
```

* Groups similar embeddings into `n_clusters`.
* If you have enough samples, each cluster = group of reports with similar symptom patterns.

Example:

* Cluster 0 → Gastrointestinal (nausea, vomiting, diarrhea)
* Cluster 1 → Neurological (dizziness, headache, fatigue)
* Cluster 2 → Cardiac (chest pain, shortness of breath)

---

### Step 4: 2D Visualization (t-SNE)

```python
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
X_tsne = tsne.fit_transform(embeddings)
df["x"], df["y"] = X_tsne[:, 0], X_tsne[:, 1]
```

* t-SNE reduces your 768D embedding vectors into **2D coordinates (x, y)**.
* That’s why you see **positive and negative numbers** — these are arbitrary but preserve relative distances.

So:

* Nearby points → Similar symptom narratives
* Distant points → Unrelated symptoms

---

### Step 5: Plot with Explainability Layers

#### (a) 🔴🟠🟢 Clusters by Severity

```python
fig = px.scatter(df, x="x", y="y", color="modifier", ...)
```

* Shows how severity (`high`, `medium`, `low`) distributes across semantic clusters.
* Example:

  * One cluster mostly red → critical/hospitalized ADEs
  * Another green → mild, transient symptoms

#### (b) 🌈 Clusters by Severity Source

```python
fig2 = px.scatter(df, x="x", y="y", color="severity_source", ...)
```

* Shows whether severity came from classifier (ML) or rule-based fallback.

#### (c) 👶🧓 Clusters by Age Group

```python
fig3 = px.scatter(df, x="x", y="y", color="age_group", ...)
```

* Helps spot **age-dependent ADE patterns**:

  * Cluster A = older patients (cardiac or severe events)
  * Cluster B = younger patients (mild fever, fatigue)

---

## 🧩 2️⃣ How to *Interpret* the Three Plots

| Plot                              | What It Shows           | How to Read It                                                              |
| --------------------------------- | ----------------------- | --------------------------------------------------------------------------- |
| **By Severity (High/Medium/Low)** | Severity distribution   | Red areas = severe clusters, green = mild clusters                          |
| **By Severity Source**            | How labels were derived | Classifier-based regions = model confident; rule-based = heuristic fallback |
| **By Age Group**                  | Age-related trends      | Can reveal age-dependent susceptibility                                     |

---

## ⚙️ 3️⃣ Practical Improvements You Can Add

1. **Label Each Cluster by Top Terms**

   ```python
   from sklearn.feature_extraction.text import CountVectorizer

   vectorizer = CountVectorizer(stop_words='english')
   X = vectorizer.fit_transform(df["entities_text"])
   terms = np.array(vectorizer.get_feature_names_out())

   for i in range(n_clusters):
       idx = df["cluster"] == i
       cluster_terms = X[idx].sum(axis=0).A1
       top_terms = terms[cluster_terms.argsort()[-5:][::-1]]
       st.write(f"🧩 Cluster {i}: {', '.join(top_terms)}")
   ```

   ➤ This tells you *what kind of symptoms* dominate each cluster.

2. **Interactive filtering** (Streamlit `multiselect`) for age, severity, or drug.

3. **Outlier detection** (e.g., HDBSCAN noise points) to highlight rare ADEs.

---

## 💡 Summary

| Concept             | Meaning                                           |
| ------------------- | ------------------------------------------------- |
| SentenceTransformer | Converts ADE/Drug text into semantic embeddings   |
| t-SNE Plot Axes     | Abstract 2D projection (distance = similarity)    |
| KMeans/HDBSCAN      | Groups similar symptom narratives                 |
| Hybrid Severity     | Merges ML + heuristic logic                       |
| Streamlit Tabs      | Visualize by Severity, Source, or Age Group       |
| Interpretation      | Helps clinicians see ADE clusters and risk trends |

---

Would you like me to add the **cluster labeling (top keywords per cluster)** and a **Streamlit dropdown to explore clusters interactively** next? That makes the visualization fully usable for doctors or analysts.

Perfect ✅ — let’s turn your **Streamlit ADEGuard pipeline** into a **formal project report / presentation-ready explanation**.

Below is a structured write-up (with sections, subheadings, and flow explanations). It’s designed so you can paste it directly into your report or slides.

---

# 🧠 ADEGuard: Hybrid Adverse Drug Event (ADE) Detection and Severity Analysis

---

## 🔷 1. Project Overview

**ADEGuard** is an intelligent text analytics system designed to automatically identify **Adverse Drug Events (ADEs)** from unstructured clinical narratives and analyze their **severity**.

It integrates three core AI capabilities into one interactive Streamlit dashboard:

1. **Named Entity Recognition (NER)** – Extract ADEs and Drug mentions from text using a BioBERT model.
2. **Severity Classification** – Categorize each ADE as *Severe*, *Moderate*, or *Mild* using a Transformer-based classifier.
3. **Explainability and Clustering** – Provide transparent explanations (via SHAP) and visualize latent ADE patterns using clustering.

This hybrid approach combines **data-driven NLP** with **explainable AI** to support clinical interpretation and pharmacovigilance.

---

## 🔷 2. System Architecture

### ⚙️ Overall Pipeline

```
   ┌─────────────────────┐
   │ Uploaded CSV (Text + Age) │
   └────────────┬────────────┘
                │
         ▼ Text Preprocessing
                │
   ┌────────────┴────────────┐
   │ BioBERT NER Model       │ → Extracts ADE & Drug entities
   └────────────┬────────────┘
                │
         ▼ Symptom Text
                │
   ┌────────────┴────────────┐
   │ Severity Classifier     │ → Predicts Mild / Moderate / Severe
   └────────────┬────────────┘
                │
         ▼ SHAP Explainability
                │
   ┌────────────┴────────────┐
   │ Clustering Module       │ → Groups ADE patterns by similarity
   └─────────────────────────┘
```

---

## 🔷 3. Data Input and Preprocessing

### 📤 CSV Upload

The user uploads a `.csv` file containing two columns:

* `symptom_text` – free-text clinical description (e.g., *“Patient experienced high fever after dose.”*)
* `age` – patient’s age.

### 👶 Age Grouping

A preprocessing function categorizes patients into:

* **Child** (<18)
* **Young Adult** (18–39)
* **Middle Age** (40–59)
* **Senior** (≥60)

This enables **demographic analysis** of ADE severity patterns.

---

## 🔷 4. ADE/Drug Named Entity Recognition (NER)

### 📘 Model

A **BioBERT Token Classification** model is used, fine-tuned on biomedical entity recognition tasks.

### ⚙️ Functionality

* Tokenizes input text using BioBERT’s tokenizer.
* Predicts token-level labels:

  * `B-ADE`, `I-ADE` for Adverse Drug Events
  * `B-DRUG`, `I-DRUG` for Drugs
  * `O` for non-entities.
* Consecutive “B/I” tokens are merged to form complete entities (e.g., *“severe rash”*).

### 🧾 Output

| symptom_text                   | age_group   | ADE   | DRUG        |
| ------------------------------ | ----------- | ----- | ----------- |
| “Fever after paracetamol dose” | Young Adult | Fever | Paracetamol |

### 💡 Visualization

Each token is color-coded:

* 🔴 **Red:** ADE tokens
* 🔵 **Blue:** Drug tokens

This provides an interpretable token-level visualization of model predictions.

---

## 🔷 5. Severity Classification

### 📘 Model

A **Transformer-based Sequence Classification** model (e.g., DistilBERT / BioClinicalBERT) fine-tuned for ADE severity.

### ⚙️ Prediction

* Input: Symptom narrative text.
* Output: Probability distribution across three classes — *Mild*, *Moderate*, *Severe*.
* Highest-probability class is chosen as the predicted label.

Example:

```
Input: "Patient hospitalized due to severe allergic reaction."
Output: Severe (Confidence: 0.98)
```

### 📊 Output Table

| symptom_text                            | pred_label |
| --------------------------------------- | ---------- |
| “High fever and chills”                 | Moderate   |
| “Slight pain at injection site”         | Mild       |
| “Anaphylaxis requiring hospitalization” | Severe     |

---

## 🔷 6. Explainability using SHAP

### 🧮 Motivation

In healthcare AI, interpretability is crucial.
SHAP (SHapley Additive exPlanations) quantifies each token’s contribution to the model’s decision.

### ⚙️ Process

1. A SHAP explainer wraps the Hugging Face pipeline.
2. For a selected text sample, SHAP computes **per-token importance values**.
3. Tokens influencing the prediction more strongly receive higher SHAP values.

### 🎨 Visualization

* Tokens are highlighted in shades of red proportional to their importance.
* A bar chart displays top influential words.

Example:

```
Text: “Severe chest pain and high fever after vaccination.”
Tokens “severe” and “chest pain” show strongest positive SHAP values.
```

### 📈 Output

* **Heatmap:** Redder tokens = stronger contribution to "Severe".
* **Bar Chart:** Word importance ranking for transparency.

---

## 🔷 7. Clustering and Pattern Discovery (Hybrid Analysis)

*(From your extended code version)*

### 🧮 Embedding Model

A **SentenceTransformer (all-MiniLM-L6-v2)** converts extracted entity text (ADE + DRUG) into dense embeddings.

### ⚙️ Clustering

* K-Means groups similar ADE/Drug embeddings.
* t-SNE reduces dimensions for visualization.
* Clusters are visualized using Plotly.

### 🎨 Color Coding

* Color = Severity (High / Medium / Low)
* Hover Info = Entity details + Age group

### 🧠 Clinical Insight

Doctors can observe clusters such as:

* **Elderly + Severe Reactions** grouped together.
* **Mild ADEs** forming a distinct region.

This helps identify population-specific ADE trends.

---

## 🔷 8. Integrated Dashboard

| **Tab**                          | **Module**                  | **Purpose**                       |
| -------------------------------- | --------------------------- | --------------------------------- |
| **1. NER**                       | BioBERT model               | Extract ADE and Drug entities     |
| **2. Severity + Explainability** | Classifier + SHAP           | Predict and explain severity      |
| **3. Clustering**                | SentenceTransformer + t-SNE | Visualize ADE similarity patterns |

The dashboard provides both **automation** and **interpretability**, bridging the gap between AI and clinical reasoning.

---

## 🔷 9. Key Strengths

| Feature                        | Description                                                          |
| ------------------------------ | -------------------------------------------------------------------- |
| **Biomedical Language Models** | Domain-tuned BioBERT & ClinicalBERT models capture medical nuances.  |
| **Explainable AI (SHAP)**      | Clinicians can see “why” a severity label was predicted.             |
| **Age-aware Analysis**         | Adds demographic insight to ADE clustering.                          |
| **Hybrid Rules + ML**          | Falls back to rule-based heuristics when model confidence is low.    |
| **Interactive Visualization**  | Streamlit UI enables token-level, severity, and cluster exploration. |

---

## 🔷 10. Potential Applications

* **Regulatory reporting** (FDA / EMA pharmacovigilance)
* **Drug safety monitoring**
* **Post-marketing surveillance**
* **Clinical decision support**
* **AI audit and transparency tools**

---

## 🔷 11. Future Enhancements

* 🧠 **Multi-task Learning:** Joint NER + Severity prediction in a single model.
* 📊 **Temporal ADE Analysis:** Track severity progression over time.
* 🩺 **Knowledge Graph Integration:** Link ADEs with known drug safety profiles.
* 🔍 **Zero-shot or few-shot models:** Adapt to unseen ADE types.

---

Would you like me to generate a **formatted project report (PDF)** version next — with headings, numbered sections, and embedded diagrams (pipeline architecture + sample visual outputs placeholders)?
It’ll be export-ready for submission or presentation slides.
