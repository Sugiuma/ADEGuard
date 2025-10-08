## 🧾 Model Card — Severity Classifier (BioBERT)
### **Model Type:** Sequence Classification (Severity Prediction) based on BioBERT

### **Intended Use**
**Primary Purpose:**
Classify the *severity level* (Mild / Moderate / Severe) of an identified Adverse Drug Event (ADE) described in free-text clinical reports or VAERS narratives.

**Applications:**
* 🔎 Prioritize high-severity ADE cases for human review.
* 🏥 Support hospital pharmacovigilance and safety signal detection.
* 📈 Input for dashboards and severity trend analytics (ADEGuard).
* 🔗 Downstream integration with ADE/DRUG NER output to create full event triplets: `(DRUG, ADE, SEVERITY)`.

**Input:**
Short free-text span or sentence describing an ADE (e.g., `"Patient experienced severe headache after Moderna booster"`).

**Output:**
Severity class prediction:
* **0 — Severe**
* **1 — Moderate**
* **2 — Mild**

### **Training Details**
**Base Model:** `dmis-lab/biobert-base-cased-v1.1`
**Framework:** Hugging Face Transformers (PyTorch)
**Training Dataset:**

* Derived from VAERS vaccine reports (2020–2025).
* Severity labels derived from *weak supervision* (Snorkel) + manual corrections.

**Weak Labels Source:**
* Snorkel LFs on structured fields: `DIED`, `HOSPITAL`, `L_THREAT`, `DISABLE`.
* Text heuristics: `"mild"`, `"moderate"`, `"severe"`,  etc.

**Training Setup:**
* Train/val/test split = 70/15/15 stratified.
* Learning rate = 2e-5, batch size = 16, epochs = 3.
* Optimizer: AdamW, Weight decay = 0.01.
* Evaluation metric: Weighted F1.


**Confusion Insights:**
* Moderate vs Mild occasionally confused (context overlap).
* Severe class stable, mainly influenced by hospitalization or death indicators.

⚠️ *Performance may vary with text domain or missing contextual cues (e.g., "rash" alone vs "rash requiring ER visit").*


### **Limitations**
* Domain-specific: Trained primarily on VAERS vaccine narratives.
* No Causality: Predicts severity, not whether drug caused ADE.
* Ambiguous Phrases: Phrases like “felt bad” or “tired” may yield uncertain predictions.
* Dependent on upstream NER: Requires correctly extracted ADE spans.
* No long-text context: Model trained on snippets (<128 tokens).


### **Ethical and Responsible Use**
* 🧑‍⚕️ Intended for *research and pharmacovigilance professionals*.
* ⚠️ *Not for clinical diagnosis or automated patient triage.*
* 🔒 Data used are *de-identified* VAERS reports in compliance with FDA/CDC policies.
* Transparency: Weak labels introduced probabilistic uncertainty—outputs should be verified by experts.


### **Model Outputs Example**
| Input                                                | Predicted Severity |
| ---------------------------------------------------- | ------------------ |
| “Patient developed mild swelling at injection site.” | Mild               |
| “Hospitalized due to severe allergic reaction.”      | Severe             |
| “Fever and headache lasted 2 days.”                  | Moderate           |



### **Intended Integration**
Part of the **ADEGuard Clinical Safety Pipeline**:


### **Citation / Acknowledgment**
> Built using `dmis-lab/biobert-base-cased-v1.1` and VAERS 2020–2025 public data.
> Weak supervision inspired by Snorkel (Ratner et al., 2020).
> Developed as part of ADEGuard for pharmacovigilance and safety monitoring.

