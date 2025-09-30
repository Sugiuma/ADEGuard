# **Project Report: ADEGuard – AI-Powered Adverse Drug Event (ADE) Detection and Analysis System**

## **1. Introduction**

Adverse Drug Events (ADEs) are a major public health concern, often hidden in unstructured symptom narratives within spontaneous reports such as VAERS. Manual review is time-consuming, error-prone, and non-scalable.
To address this, I developed **ADEGuard**, an AI-powered system that combines **Natural Language Processing (NLP)**, **weak supervision**, and **explainable ML** to automatically extract, cluster, and classify ADEs from unstructured and structured clinical data.

The goal is to support **hospitals, regulators, and pharmaceutical companies** with **real-time ADE insights**, enabling improved drug safety monitoring, faster response times, and prevention of severe complications—tailored by **patient age groups and symptom modifiers**.

---

## **2. System Objectives**

* **Extract ADE spans** from unstructured narratives using NER (ADE & DRUG mentions).
* **Cluster symptom variants** (modifier-aware, age-specific) to reveal hidden patterns.
* **Classify ADE severity** (mild, moderate, severe) using structured VAERS fields, free-text narratives, and patient metadata.

---

## **3. Data Sources**

* **Unstructured Narratives**: Free-text symptom descriptions from VAERS-like reports.
* **Structured Fields**: Symptom checkboxes, patient demographics, outcome fields.
* **Gold Data**: Manually annotated subset of reports with **ADE spans** and **DRUG spans**.
* **Weak Labels**: Derived from structured symptom fields (e.g., hospitalization = severe).

---

## **4. Methodology**

### **4.1 Data Annotation & Weak Supervision**

* **Gold Standard**: 500–1,000 manually labeled records with ADE & DRUG entities using Label Studio.
* **Weak Labels**: Generated using structured symptom metadata (e.g., "ER Visit" → moderate/severe).
---

### **4.2 ADE Span Extraction (NER)**

* Model: **BioBERT** fine-tuned for biomedical NER.
* Task: Detect **ADE spans** (B-ADE, I-ADE) and **DRUG spans** (B-DRUG, I-DRUG).
* Post-processing: Merge subword tokens, align to human-readable spans.

**Outputs:**

* Highlighted ADE/DRUG tokens in narratives.
* Exportable structured ADE–drug linkage tables.

---

### **4.3 Clustering Symptom Variants**

* Embedding Models: Sentence-BERT / BioBERT embeddings.
* Dimensionality Reduction: **t-SNE** for visualization.
* Clustering: **KMeans** for grouping similar ADEs.
* **Modifier-aware grouping**: Clusters account for severity keywords (e.g., "mild rash" vs. "severe rash").
* **Age-stratified clusters**: Separate patterns for pediatric, adult, elderly.

**Outputs:**

* Cluster plots with **hover tooltips** showing entities, age, severity.
* Discovery of **age-specific ADE subtypes**.

---

### **4.4 Severity Classification**

* Multi-input classification using:

  * **Free-text narratives** (BioBERT-based classifier).
  * **Structured fields** (hospitalization, ER visit, death).
  * **Patient metadata** (age, gender).
* Rule-based prediction

**Classes:** Mild, Moderate, Severe.

---
## **5. Streamlit UI (ADEGuard Dashboard)**

The interactive **Streamlit dashboard** includes:

1. **Token-Level NER Highlights**

   * ADE/DRUG spans highlighted in color.
   * Hover tooltips with entity metadata.

2. **Clustering Visualizations**

   * 2D scatterplots.
   * Colored by cluster, with hover info on **symptom variants + age group + severity**.

3. **Severity Classification Output**

   * Predicted label (mild/moderate/severe).
   * Supporting structured metadata.

---

## **6. Implementation Stack**

* **NLP Models**: BioBERT, Sentence-BERT.
* **Libraries**: Hugging Face Transformers, Scikit-learn, PyTorch
* **Visualization**: Streamlit, Plotly, Matplotlib.
* **Clustering**: K means, t-SNE.
* **Data Processing**: Pandas, Label Studio.

---

## **7. Impact**

* **Hospitals**: Faster triage of ADE reports by severity.
* **Regulators**: Scalable surveillance with transparent AI reasoning.
* **Pharma companies**: Post-market drug safety insights across demographics.

---

## **8. Future Enhancements**
.
* Expand beyond VAERS to global pharmacovigilance datasets.
* Use **LLM-powered reasoning** for complex ADE-drug causality detection.

---

## **9. Conclusion**

ADEGuard demonstrates that **AI + NLP + Explainable ML** can transform adverse event monitoring. By combining **span extraction, clustering, severity classification** the system provides regulators and clinicians with actionable, real-time insights to **improve drug safety and patient outcomes**.


