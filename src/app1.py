import streamlit as st
import pandas as pd
import numpy as np
import torch
import re
from snorkel.labeling import LabelModel, PandasLFApplier, labeling_function, ABSTAIN
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

st.set_page_config(layout="wide", page_title="ADE Dashboard")

# -----------------------------
# 1️⃣ Upload CSV
# -----------------------------
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV with symptom_text + AGE", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

# Ensure required column
if "symptom_text" not in df.columns or "AGE" not in df.columns:
    st.error("CSV must contain 'symptom_text' and 'AGE' columns.")
    st.stop()

# -----------------------------
# 2️⃣ NER: Extract ADE/DRUG spans
# -----------------------------
st.header("1️⃣ NER: Extract ADE/DRUG")
@st.cache_resource
def load_ner_model():
    model_name = "dmis-lab/biobert-base-cased-v1.1"  # Replace with NER model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
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
# 3️⃣ Weak Labeling (Severity)
# -----------------------------
st.header("2️⃣ Weak Labeling & Severity")

SEVERE, MODERATE, MILD = 0, 1, 2
id2label = {0:"Severe",1:"Moderate",2:"Mild"}

# Simple labeling functions
@labeling_function()
def lf_severe(row):
    text = str(row.get("symptom_text","")).lower()
    return SEVERE if any(w in text for w in ["death","critical","severe","hospitalized"]) else ABSTAIN

@labeling_function()
def lf_moderate(row):
    text = str(row.get("symptom_text","")).lower()
    return MODERATE if "moderate" in text else ABSTAIN

@labeling_function()
def lf_mild(row):
    text = str(row.get("symptom_text","")).lower()
    return MILD if "mild" in text else ABSTAIN

lfs = [lf_severe, lf_moderate, lf_mild]

if "weak_label_id" not in df.columns:
    applier = PandasLFApplier(lfs)
    L = applier.apply(df)
    label_model = LabelModel(cardinality=3, verbose=False)
    label_model.fit(L_train=L, n_epochs=300, seed=42)
    df["weak_label_id"] = label_model.predict(L)
    df["weak_label"] = df["weak_label_id"].map(id2label)

st.dataframe(df[["symptom_text","weak_label"]].head(10))

# -----------------------------
# 4️⃣ Age Bucketing
# -----------------------------
def classify_age_group(age):
    if pd.isna(age): return "Unknown"
    if age < 18: return "Child"
    if age < 60: return "Adult"
    return "Senior"

df["age_group"] = df["AGE"].apply(classify_age_group)

# -----------------------------
# 5️⃣ Clustering ADE/DRUG
# -----------------------------
st.header("3️⃣ Clustering ADE/DRUG")

model_embed = SentenceTransformer("all-MiniLM-L6-v2")

# Combine ADE + DRUG for embeddings
df["entities_text"] = df.apply(lambda row: " ".join(row["ADE"] + row["DRUG"]), axis=1)
embeddings = model_embed.encode(df["entities_text"].astype(str).tolist(), show_progress_bar=True)

n_clusters = st.slider("Number of clusters", min_value=2, max_value=15, value=5)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df["cluster"] = kmeans.fit_predict(embeddings)

# t-SNE visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(embeddings)
df["x"] = X_tsne[:,0]
df["y"] = X_tsne[:,1]

hover_text = [
    f"Entities: {ent}<br>Age: {age}<br>Severity: {sev}"
    for ent, age, sev in zip(df["entities_text"], df["age_group"], df["weak_label"])
]

fig = px.scatter(df, x="x", y="y", color="cluster", hover_name=hover_text, title="ADE/DRUG Clusters")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 6️⃣ Severity Explainability (SHAP)
# -----------------------------
st.header("4️⃣ Severity Prediction & Explainability")

# Optional: Load trained BioBERT classifier
# clf = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True, device=0)
# For demo, we will use weak_label as severity
st.dataframe(df[["symptom_text","weak_label","age_group"]].head(10))
st.info("✅ Severity is predicted using weak labels (can plug-in BioBERT classifier here for explainability/SHAP).")

