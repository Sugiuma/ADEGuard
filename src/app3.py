import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from html import escape
from fuzzywuzzy import process, fuzz
import torch,re
import plotly.express as px
from config import model_path,C_MODEL_PATH
import numpy as np
import plotly.express as px
import shap

# -----------------------------------------------------------
# Streamlit Config
# -----------------------------------------------------------
st.set_page_config(layout="wide", page_title="ADEGuard Dashboard")

st.title("🧠 ADEGuard 🧠")
st.subheader("Hybrid ADE Detection & Severity Analysis")

# -----------------------------------------------------------
# Sidebar: Upload CSV
# -----------------------------------------------------------
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload CSV with columns 'symptom_text' and 'AGE'", type=["csv"])
if uploaded_file is None:
    st.info("👈 Please upload a CSV file to start analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)

# Validate columns
if "symptom_text" not in df.columns or "age" not in df.columns:
    st.error("CSV must contain 'symptom_text' and 'AGE' columns.")
    st.stop()

if not all(col in df.columns for col in ["symptom_text", "age"]):
    st.error("CSV must contain 'symptom_text', 'age' columns")
else:
    # -----------------------------
    # 2️⃣ Age grouping
    # -----------------------------
    def age_group(age):
        try:
            age = int(age)
        except:
            return "Unknown"
        if age < 18:
            return "Child"
        elif age < 40:
            return "Young Adult"
        elif age < 60:
            return "Middle Age"
        else:
            return "Senior"

df["age_group"] = df["age"].apply(age_group)



# -----------------------------------------------------------
# Tabs
# -----------------------------------------------------------
tabs = st.tabs(["NER", "Severity Prediction & Explainability" ,"Clustering"])

# -----------------------------------------------------------
# 1️⃣ NER Tab
# -----------------------------------------------------------
with tabs[0]:
        st.subheader("ADE/DRUG Detection - Named Entity Recognition")
         # -----------------------------
        # 3️⃣ Load BioBERT NER
        # -----------------------------
      
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        label_list = ["B-ADE", "B-DRUG", "I-ADE", "I-DRUG", "O"]
        id2label = {i: label for i, label in enumerate(label_list)}

        @st.cache_data(show_spinner=False)
        def predict_entities(texts):
            all_entities, all_highlights = [], []

            for text in texts:
                tokens = re.findall(r"\w+|[^\w\s]", text)
                encoded = tokenizer(tokens, is_split_into_words=True,
                                    truncation=True, max_length=512, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(**encoded)
                    predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

                word_ids = encoded.word_ids(batch_index=0)
                pred_labels = []
                previous_word_idx = None

                for idx, word_idx in enumerate(word_ids):
                    if word_idx is None or word_idx == previous_word_idx:
                        continue
                    label = id2label[predictions[idx]]
                    pred_labels.append((tokens[word_idx], label))
                    previous_word_idx = word_idx

                # -----------------------------
                # Merge B/I tokens into entities
                # -----------------------------
                entities = {"DRUG": [], "ADE": []}
                current_entity, current_words = None, []

                for w, l in pred_labels:
                    if l.startswith("B-") or l.startswith("I-"):
                        entity_type = l.split("-")[1]
                        if current_entity == entity_type:
                            current_words.append(w)
                        else:
                            if current_entity and current_words:
                                entities[current_entity].append(" ".join(current_words))
                            current_entity = entity_type
                            current_words = [w]
                    else:
                        if current_entity and current_words:
                            entities[current_entity].append(" ".join(current_words))
                        current_entity, current_words = None, []

                if current_entity and current_words:
                    entities[current_entity].append(" ".join(current_words))

                # Direct output (no fuzzy normalization)
                all_entities.append((entities.get("ADE", []), entities.get("DRUG", [])))
                all_highlights.append(pred_labels)

            return all_entities, all_highlights

        # Run prediction
        st.info("Extracting ADE/Drug entities using BioBERT...")
        entity_results, highlights = predict_entities(df["symptom_text"].tolist())

        # Add to DataFrame
        df["ADE"], df["DRUG"] = zip(*entity_results)
        df["ADE"] = df["ADE"].apply(lambda x: ", ".join(x) if x else "None")
        df["DRUG"] = df["DRUG"].apply(lambda x: ", ".join(x) if x else "None")
        df["highlights"] = highlights

        

        # -----------------------------
        # 4️⃣ Display Final Output
        # -----------------------------
        st.success("✅ Entity Extraction Complete!")
        st.dataframe(df[["symptom_text", "age_group", "ADE", "DRUG"]].head(10))

        # -----------------------------
        # 5️⃣ Token-Level Highlight Example
        # -----------------------------
       
        st.subheader("Token-Level ADE/Drug Highlights")
        row_idx = st.number_input("Select Row Index to Highlight Tokens", min_value=0, max_value=len(df)-1, value=0)

        highlight_row = df.iloc[row_idx]
        html_text = ""
        for token, tag in highlight_row["highlights"]:
            color = "#ffcccc" if tag in ["B-ADE", "I-ADE"] else "#cce5ff" if tag in ["B-DRUG","I-DRUG"] else "white"
            html_text += f'<span style="background-color:{color};padding:2px;margin:1px;border-radius:2px;">{escape(token)}</span> '

        st.markdown(html_text, unsafe_allow_html=True)

        # -----------------------------
        # 6️⃣ Optional Download
        # -----------------------------
        if st.button("💾 Download Processed CSV"):
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_data, file_name="ade_drug_extracted.csv")


# ==================================================
# TAB 1 — SEVERITY CLASSIFIER + SHAP
# ==================================================
with tabs[1]:
    st.subheader("Severity Classification and Explainability")

    # --- Load model ---
    @st.cache_resource
    def load_classifier(path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            return tokenizer, model
        except Exception as e:
            st.error(f"❌ Error loading classifier: {e}")
            return None, None

    tokenizer, model = load_classifier(C_MODEL_PATH)

    if model and tokenizer:
        clf = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1
        )

        id2label = {0: "Severe", 1:"Moderate", 2: "Mild"}

        # --- Predict severity ---
        def predict_severity(text):
            preds = clf(text)
            scores = [d["score"] for d in preds[0]]
            pred_id = int(np.argmax(scores))
            label = id2label.get(pred_id, "Unknown")
            #label = id2label.get(int(np.argmax(scores)), "Unknown")
            return label, scores

        df["pred_label"], df["pred_scores"] = zip(*df["symptom_text"].apply(predict_severity))

        st.markdown("Classifier Predictions")
        st.dataframe(df[["symptom_text", "pred_label"]])

      # --- SHAP explainability ---
        # --- SHAP Explainability Section ---
        # SHAP explainability section (updated for Streamlit)
        # --- SHAP explainability section ---
        import shap
        import streamlit.components.v1 as components
        import numpy as np
        import pandas as pd

        @st.cache_resource(show_spinner=False)
        def get_explainer(_pipeline_model):
            return shap.Explainer(_pipeline_model)

        def st_shap(js_html, height=300):
            shap_html = f"<head>{shap.getjs()}</head><body>{js_html}</body>"
            components.html(shap_html, height=height)

        explainer = get_explainer(clf)

        row_idx = st.number_input(
            "Select Row Index for SHAP Explanation",
            min_value=0,
            max_value=len(df) - 1,
            value=0
        )

        example_text = df.iloc[row_idx]["symptom_text"]

        with st.spinner("Computing SHAP values..."):
            shap_values = explainer([example_text])

        st.success("✅ SHAP values computed")

        # Token-level explanation (raw HTML string)
        text_html = shap.plots.text(shap_values[0], display=False)  # returns str
        st_shap(text_html, height=200)

        # Tokenize and get offsets for word aggregation
        encoding = tokenizer(example_text, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        offsets = encoding["offset_mapping"]

        # Aggregate subword tokens to words
        # Aggregate subword tokens to words using mean absolute SHAP values
        word_map = []
        word_scores = []
        current_word = ""
        current_vals = []

        for tok, val, (s, e) in zip(tokens, shap_values[0].values, offsets):
            if tok in tokenizer.all_special_tokens:
                continue
            if tok.startswith("##"):
                current_word += tok[2:]
                current_vals.append(val)
            else:
                if current_word:
                    # Use mean absolute value instead of mean raw value
                    word_map.append(current_word)
                    word_scores.append(np.mean(np.abs(current_vals)))
                current_word = tok
                current_vals = [val]
        if current_word:
            word_map.append(current_word)
            word_scores.append(np.mean(np.abs(current_vals)))


        # Normalize scores (0-1)
        norm_scores = (np.array(word_scores) - np.min(word_scores)) / (np.ptp(word_scores) + 1e-6)

        # Highlighted text for UI
        highlighted_text = ""
        for word, score in zip(word_map, norm_scores):
            color = f"rgba(255,0,0,{0.3 + 0.7 * score})"
            highlighted_text += f"<span style='background-color:{color};padding:2px;margin:1px;border-radius:3px'>{word}</span> "

        st.markdown("### Token-level Importance Highlight")
        st.markdown(highlighted_text, unsafe_allow_html=True)

        # Bar chart display
        importance_df = pd.DataFrame({"Word": word_map, "Importance": word_scores})
        importance_df = importance_df.sort_values("Importance", ascending=False)
        st.subheader("Word Importance Scores")
        st.bar_chart(importance_df.set_index("Word"))


# ==================================================
# TAB 2 — CLUSTERING WITH HYBRID SEVERITY
# ==================================================
with tabs[2]:
    st.subheader("ADE/DRUG Clustering")

    # --- Sentence embeddings ---
    model_embed = SentenceTransformer("all-MiniLM-L6-v2")
    df["entities_text"] = df.apply(lambda r: " ".join(r["ADE"] + r["DRUG"]), axis=1)
    embeddings = model_embed.encode(df["entities_text"].astype(str).tolist(), show_progress_bar=True)

    # --- Classifier setup (reuse if available) ---
    @st.cache_resource
    def load_pipeline(path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            return pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            st.error(f"❌ Error loading classifier model: {e}")
            return None

    clf = load_pipeline(C_MODEL_PATH)

    # --- Hybrid severity with explainability ---
    id2label = {0: "Severe", 1:"Moderate", 2: "Mild"}
    label2level = {"severe": "high", "moderate": "medium","mild": "low"}

    def hybrid_severity_explain(text):
        """Combine classifier + rule-based fallback with explainability"""
        text_low = text.lower()

        if clf:
            try:
                preds = clf(text)
                pred_idx = int(np.argmax([d["score"] for d in preds[0]]))
                label = id2label.get(pred_idx, "Unknown").lower()
                if label in label2level:
                    return label2level[label], "classifier"
            except Exception:
                pass

        if any(w in text_low for w in ["severe", "hospitalized", "death", "critical"]):
            return "high", "rule-based"
        elif any(w in text_low for w in ["moderate", "treatment", "clinic", "significant", "persistent"]):
            return "medium", "rule-based"
        elif any(w in text_low for w in ["mild", "slight", "minor"]):
            return "low", "rule-based"

        return "unknown", "rule-based"

    # --- Apply hybrid severity ---
    results = df["symptom_text"].apply(hybrid_severity_explain)
    df[["modifier", "severity_source"]] = pd.DataFrame(results.tolist(), index=df.index)

    # --- Clustering ---
    n_samples = len(df)
    n_clusters = min(3, n_samples)
    perplexity = max(2, min(30, (n_samples - 1) / 3))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(embeddings)

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    X_tsne = tsne.fit_transform(embeddings)
    df["x"], df["y"] = X_tsne[:, 0], X_tsne[:, 1]

    # --- Hover text and plot ---
    df["hover_text"] = [
        f"Entities: {ent}<br>Age: {age}<br>Severity: {mod}<br>Source: {src}"
        for ent, age, mod, src in zip(df["entities_text"], df["age_group"], df["modifier"], df["severity_source"])
    ]

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="modifier",
        hover_name="hover_text",
        title="ADE/DRUG Clusters (Hybrid Severity with Explainability)",
        color_discrete_map={"low": "green", "medium": "orange", "high": "red"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2️⃣ Clusters by Severity
    #with col2:
    fig2 = px.scatter(
        df, x="x", y="y", color="severity_source",
        hover_name="hover_text",
        color_continuous_scale="RdYlBu",
        title="By Severity"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 3️⃣ Clusters by Age Group
    #with col3:
    fig3 = px.scatter(
        df, x="x", y="y", color="age_group",
        hover_name="hover_text",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="By Age Group"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # --- Summary table ---
    st.markdown("Cluster Summary")
    summary = df.groupby(["cluster", "modifier", "severity_source"]).size().reset_index(name="count")
    st.dataframe(summary)
    st.dataframe(df)
print(df)