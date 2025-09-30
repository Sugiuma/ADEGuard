## **How to Run**

Download the model from:
https://drive.google.com/drive/folders/1pnQJfxpMILCO2r7FUH3wotmoJ6Y2vE4F?usp=sharing


## **Steps**
1. Set up virtual environment.
   
2. Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```
2a. Update your model_path in config file.

3. Run predictions

```bash
python inference.py
```
   
4. Run inference on a CSV batch_inference:

```bash
python batch_inference.py --input_csv batch_input.csv --output_csv predicted_entities.csv
```

* **`input.csv`** must have a column named `symptom_text`.
* The script outputs **`predicted_entities.csv`** with two new columns: `predicted_DRUG` and `predicted_ADE`.

5. Run the Streamlit app

```bash
streamlit run app.py
```
   
Upload sample1.csv conatining `symptom_text`, `age`, `severity columns`. (Check for sample in `data` folder in this repo)

**Model Performance Reports**
[Metrics](https://github.com/Sugiuma/ADEGuard/blob/main/docs/training.png)
[Classification Report](https://github.com/Sugiuma/ADEGuard/blob/main/docs/classif_report.png)
[Project Report](https://github.com/Sugiuma/ADEGuard/blob/main/docs/Project_Report.md)

[Model Card](https://github.com/Sugiuma/ADEGuard/blob/main/docs/Model_Card.md)
[Project pipeline](https://github.com/Sugiuma/ADEGuard/blob/main/docs/Pipeline.md)


