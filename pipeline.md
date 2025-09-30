
# **ADEGuard – Streamlit-Based ADE Analysis Pipeline**

## **1. Overview**

The Streamlit dashboard serves as an **interactive interface** for ADEGuard, allowing users to upload datasets, run **NER extraction**, cluster symptom variants, classify severity, and visualize results. It supports **token-level highlights**, **age- and severity-aware clustering**, and **downloadable post-processed outputs**.

---

## **2. Dataset Upload**

* Users upload a **CSV file** containing at least the following columns:

  * `symptom_text` (free-text narrative)
  * `age` (numeric)
  * `severity` (raw severity label or description)
* The uploaded CSV is **previewed** for quality checks.
* Validation ensures required columns exist; otherwise, the user is notified.

---

## **3. Age Grouping**

* The pipeline categorizes patients into **age groups**:

  * `Child` (<18 years)
  * `Young Adult` (18–39)
  * `Middle Age` (40–59)
  * `Senior` (60+)
* This stratification enables **age-specific clustering and insights**.

---

## **4. ADE / DRUG Extraction (NER)**

* Users can enable **BioBERT NER**, which identifies **ADE** and **DRUG spans** in free-text narratives.

* **Post-processing** steps:

  1. Merge subword tokens into coherent entity spans.
  2. Use **fuzzy matching** against reference dictionaries (`DRUG_DICT`, `ADE_DICT`) for normalization.

* The outputs:

  * `predicted_entities`: concatenated ADE/DRUG strings for clustering
  * `highlights`: token-level labels for visualization
  * `post_entities`: cleaned entity dictionaries for each record

* If BioBERT is disabled, the raw text is used as a fallback.

---

## **5. Severity Classification**

* Severity is assigned using **both narrative text and structured severity column**:

  * Severe: contains keywords like `"severe"`, `"hospitalized"`, or `"death"`
  * Moderate: contains `"moderate"`
  * Mild: all other cases
* Result stored in `severity_label` column.

---

## **6. Embeddings & Clustering**

* **SentenceTransformer embeddings** (`all-MiniLM-L6-v2`) are generated from predicted entities.
* Users can choose **number of clusters (K)** for **KMeans clustering**.
* Each record receives a cluster ID in the `cluster` column.

---

## **7. Dimensionality Reduction**

* **t-SNE** is applied to embedding vectors to reduce dimensionality for visualization (`x` and `y` coordinates).
* This allows **interactive scatterplots** to show clusters in 2D space.

---

## **8. Cluster Visualization**

* Three types of **Plotly scatterplots** are displayed:

  1. **By severity label** (color-coded with continuous scale)
  2. **By age group** (categorical color palette)
  3. **Hover tooltips** display:

     * Predicted ADE/DRUG entities
     * Age group
     * Severity label
* This enables **quick identification of patterns**, e.g., clusters of severe ADEs in seniors.

---

## **9. Token-Level Highlighting**

* Users select a row index to view **token-level highlights**:

  * ADE tokens → red background
  * DRUG tokens → blue background
  * Other tokens → white background
* This provides **explainability at the narrative level** and aids validation of NER predictions.

---

## **10. Post-Processed Table & Download**

* A summary table shows:

  * `predicted_entities`
  * `post_entities`
  * Cluster assignment
  * Severity label
  * Age group
* Users can **download the post-processed dataset** for further analysis or reporting.

---

## **11. Pipeline Summary**

1. Upload CSV → validate columns
2. Assign age groups
3. Run NER extraction with optional BioBERT
4. Assign severity labels
5. Generate embeddings and cluster ADE/DRUG entities
6. Reduce dimensions for visualization (t-SNE)
7. Display interactive cluster plots (severity, age)
8. Highlight tokens for individual narratives
9. Show table of post-processed data with option to download


Do you want me to do that?
