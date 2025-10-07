### **BIOBERT NER Model Training**

| **Step** | **Component**                     | **Summary**                                                                                                                                                                       |
| -------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1️⃣      | **Gold Data Creation**            | Manually annotated ADE and DRUG entities from clinical narratives to build a high-quality labeled dataset using Label Studio.                                                                        |
| 2️⃣      | **Weak Supervision Augmentation** | Integrated additional weakly labeled data to expand training coverage.                                                                      |
| 3️⃣      | **Class Weight Balancing**        | Computed **class weights** to counter label imbalance — ensuring the model learns equally across frequent and rare entity types.                                                  |
| 4️⃣      | **Layer Freezing Strategy (Transfer Learning)**       | Used **progressive fine-tuning**: froze lower layers of BioBERT and **unfroze last 4 encoder layers** + classifier head to retain domain knowledge while adapting to ADE context. |
| 5️⃣ | **Training & Validation** |  Fine-tuned BioBERT on gold + weak data with weighted loss and token-level evaluation (precision, recall, F1).|
| 6️⃣   | **Post-Processing Dictionary**    | Added domain dictionary for normalization and missed-entity recovery after model inference, ensuring coverage for known ADE/Drug names.                                           |

✅ In short:
This pipeline combines weak supervision, class balancing, and selective fine-tuning to adapt BioBERT efficiently for clinical NER, while dictionary-based post-processing ensures robust entity coverage.


### **Severity Classifier Training (Snorkel Weak Supervision)**

| **Step** | **Component**                  | **Summary**                                                                                                                                                                                                          |
| -------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1️⃣      | **Input Data**                 | Used unstructured symptom_text and structured fields from VAERS dataset as input.                                                                                                                            |
| 2️⃣      | **Labeling Functions (LFs)**   | Defined **rule-based labeling functions** on structured fields (`DIED`, `HOSPITAL`, `L_THREAT`, `DISABLE`) and unstructured text (`SYMPTOM_TEXT`). Each LF votes for a class: **Severe**, **Moderate**, or **Mild**. |
| 3️⃣      | **Weak Label Generation**      | Combined multiple LF outputs using **Snorkel’s LabelModel**, which estimates LF accuracies and correlations to produce **probabilistic weak labels**.                                                                |
| 4️⃣      | **Label Aggregation**          | Converted Snorkel probabilities into **final weak labels** (`weak_label_id`, `weak_label`) and appended them to the dataset.                                                                                         |
| 5️⃣      | **Training Data Creation**     | Produced a unified dataset with **entity spans + weak severity labels** for training the downstream **severity classifier**.                                            |
| 6️⃣      | **Integration for Classifier** | These weak labels served as **pseudo-gold labels** to train a **BioBERT-based severity classifier**, reducing manual annotation effort.                                                                              |

✅ **In short:**
Snorkel was used to automatically infer **severity levels** (Severe, Moderate, Mild) from mixed structured + unstructured data, creating **weakly supervised training data** for the classifier — a scalable alternative to manual labeling.


###  **Named Entity Recognition (NER) Module**

| Step | Process                       | Description                                                    |
| ---- | ----------------------------- | -------------------------------------------------------------- |
| 1    | **Input Loading**             | Upload CSV with `symptom_text` and `age`                       |
| 2    | **Preprocessing**             | Group patients by age (Child, Young Adult, Middle Age, Senior) |
| 3    | **Model Loading**             | Load fine-tuned **BioBERT NER** model                          |
| 4    | **Tokenization & Prediction** | Split text → predict token labels (B/I-ADE, B/I-DRUG, O)       |
| 5    | **Entity Merging**            | Combine sub-tokens into complete ADE/Drug phrases              |
| 6    | **Visualization**             | Highlight ADE (red) and Drug (blue) tokens in Streamlit        |
| 7    | **Export**                    | Save enriched CSV with extracted entities                      |

---

###  **Severity Classification + Explainability**

| Step | Process                      | Description                                                       |
| ---- | ---------------------------- | ----------------------------------------------------------------- |
| 1    | **Model Loading**            | Load transformer-based severity classifier (Severe/Moderate/Mild) |
| 2    | **Prediction**               | For each symptom, predict severity and confidence scores          |
| 3    | **Hybrid Labeling**          | Apply rule-based logic for missing or ambiguous labels            |
| 4    | **Explainability (SHAP)**    | Compute SHAP values for each token to explain model’s decision    |
| 5    | **Token-Level Highlights**   | Display color-coded tokens based on importance contribution       |
| 6    | **Feature Importance Chart** | Bar chart showing key symptom words influencing severity          |

---

###  **Clustering & Visuals**

| Step | Process                      | Description                                                                     |
| ---- | ---------------------------- | ------------------------------------------------------------------------------- |
| 1    | **Sentence Embeddings**      | Convert each record (ADE + Drug + Severity) to vector using SentenceTransformer |
| 2    | **Dimensionality Reduction** | Apply **t-SNE** for 2D visualization of embeddings                              |
| 3    | **K-Means Clustering**       | Group similar ADE reports into clusters                                         |
| 4    | **Cluster Interpretation**   | Analyze grouping by **Severity**, **Age Group**, and **Drugs**                  |
| 5    | **Visualization**            | Interactive 2D scatter plots using **Plotly** for clinical insights             |



### 🩺 **Clinical Insights Dashboard (Streamlit UI)**

| **Step** | **Component**                   | **Summary**                                                                                                                                                                      |
| -------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1️⃣      | **Purpose**                     | Provides clinicians with an **interactive dashboard** to explore AI-classified **adverse drug events (ADEs)** across drugs, symptoms, severity levels, and patient demographics. |
| 2️⃣      | **Data Preparation**            | Expanded list columns (`DRUG`, `ADE`) to ensure each ADE–drug pair is analyzed individually for accurate aggregation and visualization.                             |
| 3️⃣      | **Filtering**                   | Added **dynamic filters** for **Drug**, **ADE**, and **Cluster** to explore focused subsets of the dataset interactively.                                                        |
| 4️⃣      | **Visualization**               | Displayed multiple **bar charts** showing the **distribution of severity** (`pred_label`) and **age groups**, enabling trend analysis across demographics.                       |
| 5️⃣      | **Clinical Summary Generation** | Computed a grouped **clinical summary table** (`DRUG`, `ADE`, `count`) to summarize frequent ADE–drug associations.                                                              |
| 6️⃣      | **Data Export**                 | Enabled **CSV downloads** (Filtered Cases & Clinical Summary) from the sidebar for further medical review and audit.                                                             |

✅ **In short:**
The **Clinical ADE Insights Dashboard** transforms AI model outputs into **actionable clinical intelligence**, allowing doctors and researchers to visually explore severity trends, ADE frequency, and patient demographics — bridging the gap between AI predictions and clinical interpretation.

### **Classifier Training Summary**

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
| ----- | ------------- | --------------- | -------- | -------- |
| 1     | 0.3282        | 0.1849          | 94.25%   | 93.99%   |
| 2     | 0.1664        | 0.1932          | 94.25%   | 94.13%   |
| 3     | 0.1352        | 0.2115          | 94.43%   | 94.29%   |

**Test Set Performance:**

* Loss: 0.2307
* Accuracy: 94.18%
* F1 Score: 94.08%

**Interpretation:**

* Training loss decreases, showing effective learning.
* Slight validation loss rise in later epochs but accuracy remains high → good generalization.
* Model performs reliably for ADE severity classification.


### **NER Training Summary**

| Step | Training Loss | Validation Loss | Precision | Recall | F1 Score |
| ---- | ------------- | --------------- | --------- | ------ | -------- |
| 500  | 0.0652        | 0.0287          | 75.43%    | 98.93% | 85.60%   |
| 1000 | 0.0327        | 0.0159          | 86.38%    | 99.52% | 92.48%   |
| 1500 | 0.0143        | 0.0115          | 91.99%    | 99.56% | 95.63%   |
| 2000 | 0.0114        | 0.0114          | 89.89%    | 99.81% | 94.59%   |
| 2500 | 0.0072        | 0.0099          | 92.66%    | 99.73% | 96.07%   |
| 3000 | 0.0182        | 0.0099          | 95.48%    | 99.78% | 97.58%   |
| 3500 | 0.0063        | 0.0076          | 94.81%    | 99.75% | 97.22%   |
| 4000 | 0.0045        | 0.0099          | 96.15%    | 99.82% | 97.95%   |
| 4500 | 0.0053        | 0.0088          | 96.46%    | 99.81% | 98.11%   |

**Test Set Performance:**

* Loss: 0.00564
* Precision: 96.27%
* Recall: 99.75%
* F1 Score: 97.98%

**Interpretation:**

* Training and validation loss steadily decrease → good learning.
* Precision and F1 increase over steps → the model correctly identifies ADE/Drug entities.
* High recall indicates very few ADE/Drug entities are missed.



