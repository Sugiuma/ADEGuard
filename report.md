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

Perfect 🔥 — that’s exactly how to make your project **clinically interpretable and presentation-ready**.
Let’s add a **new “📊 Clinical Insights Dashboard” tab** in your existing Streamlit app.
This tab will:

1. Let clinical reviewers **explore results interactively (filter by drug, ADE, cluster, etc.)**,
2. Display **severity and age-group distribution charts**, and
3. Provide **download buttons** for both **raw clustered data** and **clinical summary data**.

---

## ✅ Complete Streamlit Code for Clinical Insights Tab

Below is a ready-to-paste section to add **after your clustering section** in your app.

It assumes your DataFrame `df` already has these columns:
`["symptom_text","age","age_group","ADE","DRUG","pred_label","cluster"]`

---

```python
import streamlit as st
import pandas as pd

# --- Create Clinical Summary ---
def generate_clinical_summary(df):
    summary = (
        df.groupby(["DRUG", "ADE", "pred_label", "age_group"])
          .size()
          .reset_index(name="case_count")
    )
    return summary

# --- New Tab Layout ---
tabs = st.tabs(["🧩 Clustering Results", "📊 Clinical Insights Dashboard"])

# --- TAB 1 (existing): your clustering visualization code ---
with tabs[0]:
    st.write("Your clustering visualizations and explainability go here.")

# --- TAB 2: Clinical Insights Dashboard ---
with tabs[1]:
    st.header("📊 Clinical ADE Insights Dashboard")
    st.markdown("""
    This dashboard helps clinical teams explore AI-classified adverse drug events 
    by drug, symptom, severity, and age group.
    """)

    # --- Filters ---
    col1, col2, col3 = st.columns(3)
    selected_drug = col1.selectbox("Select a Drug", ["All"] + sorted(df["DRUG"].dropna().unique().tolist()))
    selected_ade = col2.selectbox("Select an ADE", ["All"] + sorted(df["ADE"].dropna().unique().tolist()))
    selected_cluster = col3.selectbox("Select Cluster", ["All"] + sorted(df["cluster"].unique().tolist()))

    filtered_df = df.copy()
    if selected_drug != "All":
        filtered_df = filtered_df[filtered_df["DRUG"] == selected_drug]
    if selected_ade != "All":
        filtered_df = filtered_df[filtered_df["ADE"] == selected_ade]
    if selected_cluster != "All":
        filtered_df = filtered_df[filtered_df["cluster"] == selected_cluster]

    # --- Show Data Preview ---
    st.subheader("📄 Filtered Case Details")
    st.dataframe(filtered_df[["symptom_text", "age", "age_group", "ADE", "DRUG", "pred_label", "cluster"]].head(20))

    # --- Charts ---
    st.subheader("📈 Severity Distribution")
    st.bar_chart(filtered_df["pred_label"].value_counts())

    st.subheader("👥 Age Group Distribution")
    st.bar_chart(filtered_df["age_group"].value_counts())

    # --- Clinical Summary Generation ---
    clinical_summary = generate_clinical_summary(df)

    st.subheader("📋 Clinical Summary Table")
    st.dataframe(clinical_summary)

    # --- Sidebar Download Buttons ---
    st.sidebar.markdown("### 📥 Download Clinical Data")

    # Prepare Raw Clustered Data
    raw_csv = df[["symptom_text","age","age_group","ADE","DRUG","pred_label","cluster"]].to_csv(index=False).encode("utf-8")

    # Prepare Clinical Summary
    summary_csv = clinical_summary.to_csv(index=False).encode("utf-8")

    st.sidebar.download_button(
        label="⬇️ Download Raw Clustered Data (CSV)",
        data=raw_csv,
        file_name="clustered_raw_data.csv",
        mime="text/csv"
    )

    st.sidebar.download_button(
        label="⬇️ Download Clinical Summary (CSV)",
        data=summary_csv,
        file_name="clinical_summary.csv",
        mime="text/csv"
    )

    st.sidebar.success("✅ Clustering complete — Data ready for clinical analysis!")
```

---

## 🧠 What This Adds

| Feature                                  | Purpose                                      |
| ---------------------------------------- | -------------------------------------------- |
| **📊 Tab “Clinical Insights Dashboard”** | Dedicated view for clinical reviewers        |
| **Filters (Drug, ADE, Cluster)**         | Interactive exploration                      |
| **Bar charts**                           | Severity & age distribution overview         |
| **Summary Table**                        | Aggregated data for risk signal detection    |
| **Two sidebar downloads**                | 1️⃣ Raw Clustered Data, 2️⃣ Clinical Summary |
| **Streamlit layout**                     | Looks professional and presentation-ready    |

---

## 🩺 Example Workflow for Clinical Teams

1. **Select a Drug** → e.g., *Ibuprofen*
2. See **which ADEs** are linked and how severe they are
3. Review **age-specific distributions**
4. **Download** both CSVs:

   * `clustered_raw_data.csv` → for record-level audit
   * `clinical_summary.csv` → for reporting or Power BI

---

Would you like me to extend this further so each cluster automatically gets a **“theme description”** (e.g., Cluster 0 → “Gastrointestinal symptoms”)?
That would make the dashboard even more clinically intuitive.
Absolutely ✅ — here’s a **clear, concise summary** of the exact **steps performed in your code** for **weak label generation using Snorkel** 👇

---

## 🧩 **Steps for Weak Label Generation**

1. **Define Label Space**

   * Created integer label IDs for three severity levels:
     `SEVERE = 0`, `MODERATE = 1`, `MILD = 2`, `ABSTAIN = -1`.
   * Defined a mapping:

     ```python
     id2label = {0: "Severe", 1: "Moderate", 2: "Mild"}
     ```

2. **Design Labeling Functions (LFs)**

   * Implemented multiple **heuristic labeling functions** using Snorkel’s `@labeling_function()` decorator.
   * LFs capture both **structured features** and **text-based cues**:

     * `lf_died`, `lf_hospital`, `lf_l_threat`, `lf_disable` → label **Severe** based on serious clinical outcomes (`DIED`, `HOSPITAL`, etc.).
     * `lf_text_severe`, `lf_text_moderate`, `lf_text_mild` → label based on **keywords** in `SYMPTOM_TEXT`.

3. **Apply Labeling Functions**

   * Loaded raw data (`ade_gold_subset_20k.csv`).
   * Used `PandasLFApplier` to apply all labeling functions on each row:

     ```python
     L_train = applier.apply(df)
     ```
   * Produced a **label matrix** (`L_train`) where each row corresponds to a data sample and each column to an LF’s label output.

4. **Train Snorkel Label Model**

   * Initialized and trained Snorkel’s **LabelModel** to estimate the true label distribution:

     ```python
     label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=42)
     ```
   * The LabelModel learns **LF accuracies and correlations**, generating probabilistic labels.

5. **Generate Weak Labels & Probabilities**

   * Computed **class probabilities** (`predict_proba`) for each sample.
   * Assigned **hard labels** using `predict`.
   * Appended new columns to the DataFrame:

     * `weak_label_prob_SEVERE`, `weak_label_prob_MODERATE`, `weak_label_prob_MILD`
     * `weak_label_id`, `weak_label`

6. **Save Final Dataset**

   * Exported the enriched dataset (original data + weak labels + probabilities) to:

     ```bash
     dataset_with_weaklabels.csv
     ```

---

✅ **In summary:**

> Defined domain-specific labeling rules → Applied them to raw ADE data → Trained Snorkel’s LabelModel to combine noisy signals → Generated probabilistic and hard weak labels for severity classification.

---

Would you like me to format this into a short “Methods” section (2–3 sentences) suitable for your ADEGuard paper/report?
