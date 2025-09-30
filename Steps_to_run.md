## ** How to Run**

Download the model from:
https://drive.google.com/drive/folders/1pnQJfxpMILCO2r7FUH3wotmoJ6Y2vE4F?usp=sharing


## **5️⃣ How to Run**
1. Set up virtual environment.
   
2. Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```
3. python inference.py
   
4. Run inference on a CSV batch_inference:

```bash
ython batch_inference.py --input_csv batch_input.csv --output_csv predicted_entities.csv
```

* **`input.csv`** must have a column named `symptom_text`.
* The script outputs **`predicted_entities.csv`** with two new columns: `predicted_DRUG` and `predicted_ADE`.

