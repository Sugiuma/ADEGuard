
# **ADEGuard BioBERT NER Model Card**

**Model Name:** ADEGuard-BioBERT-NER

**Version:** 1.0

**Authors / Maintainers:** Suganya G/CureviaAI

**Model Type:** Token classification (NER) based on BioBERT

---

## **Intended Use**

* **Primary Purpose:** Extract Adverse Drug Events (ADEs) and drug mentions from free-text symptom narratives in VAERS reports or similar clinical text.
* **Applications:**

  * Pharmacovigilance and drug safety monitoring.
  * Real-time ADE surveillance for hospitals, regulators, and pharmaceutical companies.
  * Input to downstream clustering, severity classification, and explainable dashboards (ADEGuard).
* **Input:** Free-text symptom descriptions.
* **Output:** Token-level ADE/DRUG predictions, post-processed entity spans.

---

## **Limitations**

* **Domain-specific:** Trained on VAERS vaccine reports (2020–2025); performance may degrade on non-vaccine ADEs or very different clinical text.
* **NER Errors:** May miss rare ADEs or drugs not in the training vocabulary.
* **Severity Not Included:** Model only extracts entities; severity classification is handled separately via rule-based methods.
* **No Causality Detection:** The model identifies mentions but cannot confirm causal relationship between drug and ADE.
* **Post-processing Reliance:** Fuzzy matching and dictionary-based normalization improve accuracy but may introduce false positives.
* **Long Text Handling:** Input truncated to 512 tokens due to BioBERT limits; very long narratives may lose context.

---

## **Performance**

* **Token-level NER metrics (VAERS validation subset):**

  * **ADE F1-score:** ~0.88
  * **DRUG F1-score:** ~0.91
* **Entity-level accuracy:** Post-processing improved matching to known drug/ADE dictionaries.
* **Clustering & downstream tasks:** Embeddings + clustering revealed age- and modifier-aware symptom patterns.

> ⚠️ Performance varies with domain, text quality, and presence of unseen drugs/ADEs.

---

## **Ethical Considerations**

* **Intended for professional use** by clinicians, regulators, or pharmacovigilance teams.
* **Not a diagnostic tool:** Outputs should be interpreted by experts; misinterpretation may lead to inappropriate decisions.
* **Data Privacy:** Model should be used on de-identified clinical reports or VAERS data in compliance with regulations.

---

## **Training Data**

* VAERS vaccine reports (2020–2025) with gold-annotated ADE/DRUG spans.
* Augmented with weak labels from structured symptom fields.

---

