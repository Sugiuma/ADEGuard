### 🧠 **BIOBERT NER Model Training**

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


### ⚙️ **Severity Classifier Training (Snorkel Weak Supervision)**

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


