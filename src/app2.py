import streamlit as st
import pandas as pd
import numpy as np
import torch
import shap
from snorkel.labeling import LabelModel, PandasLFApplier, labeling_function, ABSTAIN
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

st.set_page_config(layout="wide", page_title="ADE Dashboard")

# -----------------------------
# Sidebar: Upload CSV
# -----------------------------
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 'symptom_text' and 'AGE'", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

# Check required columns
if "symptom_text" not in df.columns or "AGE" not in df.columns:
    st.error("CSV must contain 'symptom_text' and 'AGE' columns.")
    st.stop()

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["NER", "Clustering", "Severity Prediction", "Explainability"])

# -----------------------------
# 1️⃣ NER Tab
# -----------------------------
with tabs[0]:
    st.header("NER: Extract ADE/DRUG")

    @st.cache_resource
    def load_ner_model():
        model_name = "dmis-lab/biobert-base-cased-v1.1"  # Replace with NER model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        return ner_pipe

    ner_pipe = load_ner_model()

    def extract_entities(text):
        ner_results = ner_pipe(text)
        adrs, drugs = [], []
        for res in ner_results:
            if res['entity_group'] in ["ADE","ADR"]:  # adjust labels
                adrs.append(res['word'])
            elif res['entity_group'] in ["DRUG"]:
                drugs.append(res['word'])
        return adrs, drugs

    df["ADE"], df["DRUG"] = zip(*df["symptom_text"].apply(extract_entities))
    st.dataframe(df[["symptom_text","ADE","DRUG"]].head(10))

# -----------------------------
# 2️⃣ Clustering Tab
# -----------------------------
with tabs[1]:
    st.header("Clustering ADE/DRUG")

    model_embed = SentenceTransformer("all-MiniLM-L6-v2")
    df["entities_text"] = df.apply(lambda row: " ".join(row["ADE"] + row["DRUG"]), axis=1)
    embeddings = model_embed.encode(df["entities_text"].astype(str).tolist(), show_progress_bar=True)

    # Age bucketing
    def classify_age_group(age):
        if pd.isna(age): return "Unknown"
        if age < 18: return "Child"
        if age < 60: return "Adult"
        return "Senior"

    df["age_group"] = df["AGE"].apply(classify_age_group)

    n_clusters = st.slider("Number of clusters", min_value=2, max_value=15, value=5)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(embeddings)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(embeddings)
    df["x"] = X_tsne[:,0]
    df["y"] = X_tsne[:,1]

    hover_text = [
        f"Entities: {ent}<br>Age: {age}" for ent, age in zip(df["entities_text"], df["age_group"])
    ]

    fig = px.scatter(df, x="x", y="y", color="cluster", hover_name=hover_text, title="ADE/DRUG Clusters")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 3️⃣ Severity Prediction Tab
# -----------------------------
with tabs[2]:
    st.header("Severity Prediction (BioBERT)")

    # -----------------------------
    # Load BioBERT classifier
    # -----------------------------
    @st.cache_resource
    def load_classifier(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        clf_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True,
                            device=0 if torch.cuda.is_available() else -1)
        return clf_pipe

    classifier_model_path = "path_to_your_trained_biobert_classifier"  # Replace
    clf = load_classifier(classifier_model_path)

    # Weak labels fallback if needed
    SEVERE, MODERATE, MILD = 0, 1, 2
    id2label = {0:"Severe",1:"Moderate",2:"Mild"}

    # Predict severity
    def predict_severity(text):
        preds = clf(text)
        scores = [d["score"] for d in preds[0]]
        pred_idx = int(np.argmax(scores))
        label = id2label[pred_idx]
        confidence = scores[pred_idx]
        return label, confidence

    df["predicted_severity"], df["confidence"] = zip(*df["symptom_text"].apply(predict_severity))
    st.dataframe(df[["symptom_text","predicted_severity","confidence"]].head(10))

# -----------------------------
# 4️⃣ Explainability Tab
# -----------------------------
with tabs[3]:
    st.header("SHAP Explainability for Severity")

    # Prepare data for SHAP
    sample_texts = df["symptom_text"].tolist()[:10]  # subset for demo
    tokenized = clf.tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")

    explainer = shap.Explainer(clf.model, masker=tokenized["input_ids"])
    shap_values = explainer(tokenized["input_ids"])

    st.info("Displaying SHAP values for first 5 samples")
    for i, text in enumerate(sample_texts[:5]):
        st.write(f"Text: {text}")
        st_shap = shap.plots.text(shap_values[i])
