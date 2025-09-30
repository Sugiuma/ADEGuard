# **ADEGuard – Data & Methodology Workflow**

## **1. Data Preparation**

* **Source Data:** VAERS dataset (Vaccine Adverse Event Reporting System).
* **Subset Selection:** Combined all reports, then **filtered COVID-19 vaccine–related cases** as the primary working dataset (largest, most recent, most relevant).
* **Stratified Sampling:** To avoid skew (COVID dominated recent years),  sampled across:

  * Vaccine type
  * Age groups (pediatric, adult, elderly)
  * Outcomes (ER visits, hospitalization, death)
    → Ensured a **balanced representative mix** of ADEs.

---

## **2. Annotation (Gold Data Creation)**

* Used **Label Studio** to annotate a stratified subset.
* Task: Mark **ADE spans** (symptom phrases) and **DRUG spans** (medications/vaccines).
* Output: BIO tagging scheme (B-ADE, I-ADE, B-DRUG, I-DRUG, O).
* This gold data was used for **BioBERT fine-tuning** and evaluation.

---

## **3. Weak Supervision**

* VAERS has **structured symptom fields** (`SYMPTOM1` … `SYMPTOM5`).
*  converted these fields into **weak labels**:

  * If `SYMPTOM1` = “Headache” → label narrative text spans containing “headache” as ADE.
  * If drug/vaccine name is recorded, map to `DRUG` entity.
* These weak labels were combined with gold annotations for **semi-supervised training**.

---

## **4. Handling Imbalanced Data**

* **Problem:** Severe ADEs (death, hospitalization) are much rarer than mild ones.
* **Solution:**

  * Used **class weights** in severity classifier loss function to penalize underrepresented classes more.
  * Stratified sampling ensured minority classes appear in both training & validation.
  * Evaluated with **macro-averaged metrics** (balanced across classes).

---

## **5. ADE Extraction (NER)**

* Fine-tuned **BioBERT** on annotated + weak-labeled data.
* Output: ADE spans and DRUG spans from narratives.
* Post-processing merged subword tokens for clean phrase-level ADE detection.

---

## **6. Clustering**

* Extracted embeddings from **Sentence-BERT** for symptom spans.
* Applied **t-SNE** for dimensionality reduction.
* Used **K means** to discover clusters.
* **Modifier-aware & Age-specific:**

  * Severity words (“mild”, “severe”) included in embeddings.
  * Age-stratified clustering revealed differences (e.g., “seizure” clusters in children vs. “chest pain” in elderly).

---

## **7. Severity Classification**

* Multi-input model combining:

  * **Narrative embeddings** (BioBERT CLS token).
  * **Structured fields** (hospitalization flag, ER visit, death outcome).
  * **Patient metadata** (age, gender).
* **Rule based severity classifier** outputs: Mild / Moderate / Severe.


---

## **8. Streamlit UI (ADEGuard Dashboard)**

The pipeline was integrated into a **Streamlit app** with:

* **Token Highlighting:** ADE/DRUG spans color-coded in narrative text.
* **Cluster Visualization:** Interactive 2D plots (t-SNE), with **hover text** showing ADE, age group, severity.
* **Severity Prediction:** Displays predicted label with structured metadata context.

---

## **9. Prediction on New Data**

* **CSV Upload:** Users can upload raw narrative.
* Pipeline automatically:

  1. Runs NER → extract ADE/DRUG spans.
  2. Classifies severity.
  3. Places symptoms into clusters.
  4. Visualizes results in UI (highlighted tokens, cluster map, severity tag).
* Designed to scale for **batch processing** of new safety reports.



# **Streamlit FLow**

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


