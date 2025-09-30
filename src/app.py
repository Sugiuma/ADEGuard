import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from html import escape
from fuzzywuzzy import process, fuzz
import torch,re
import plotly.express as px
from config import model_path


st.title("ADEGuard – AI-Powered (ADE) Detection & Analysis")

# -----------------------------
# 1️⃣ Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    if not all(col in df.columns for col in ["symptom_text", "age", "severity"]):
        st.error("CSV must contain 'symptom_text', 'age', and 'severity' columns")
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

        

        # -----------------------------
        # 3️⃣ BioBERT NER + Post-Processing
        # -----------------------------
        use_biobert = st.checkbox("Use BioBERT NER for ADE/Drug extraction", value=True)

        if use_biobert:
            st.info("Loading BioBERT NER model...")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            label_list = ["B-ADE","B-DRUG", "I-ADE", "I-DRUG", "O"]
            id2label = {i: label for i, label in enumerate(label_list)}

            # Reference drug/ADE dictionary for fuzzy matching
            DRUG_DICT = ["pfizer", "moderna", "astrazeneca", "covaxin", "janssen", "biontech"]
            ADE_DICT = ["dizziness", "headache", "fatigue", "rash", "nausea", "fever", "itching", "chest pain"]

            from fuzzywuzzy import process, fuzz

            @st.cache_data
            def predict_entities(texts):
                results, highlights, post_entities = [], [], []
                for text in texts:
                    # Tokenize
                    tokens = re.findall(r"\w+|[^\w\s]", text)
                    encoded = tokenizer(tokens, is_split_into_words=True, truncation=True,
                                        max_length=512, return_tensors="pt").to(device)

                    with torch.no_grad():
                        outputs = model(**encoded)
                        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

                    # Align predictions to words
                    word_ids = encoded.word_ids(batch_index=0)
                    pred_labels = []
                    previous_word_idx = None
                    for idx, word_idx in enumerate(word_ids):
                        if word_idx is None or word_idx == previous_word_idx:
                            continue
                        pred_labels.append((tokens[word_idx], id2label[predictions[idx]]))
                        previous_word_idx = word_idx

                    # -----------------------------
                    # Post-processing / entity merging
                    # -----------------------------
                    entities = {"DRUG": [], "ADE": []}
                    current_entity, current_words = None, []
                    for w, l in pred_labels:
                        if l in ["B-DRUG", "I-DRUG"]:
                            if current_entity == "DRUG":
                                current_words.append(w)
                            else:
                                if current_entity and current_words:
                                    entities[current_entity].append(" ".join(current_words))
                                current_entity = "DRUG"
                                current_words = [w]
                        elif l in ["B-ADE", "I-ADE"]:
                            if current_entity == "ADE":
                                current_words.append(w)
                            else:
                                if current_entity and current_words:
                                    entities[current_entity].append(" ".join(current_words))
                                current_entity = "ADE"
                                current_words = [w]
                        else:
                            if current_entity and current_words:
                                entities[current_entity].append(" ".join(current_words))
                            current_entity, current_words = None, []
                    if current_entity and current_words:
                        entities[current_entity].append(" ".join(current_words))

                    # -----------------------------
                    # Fuzzy matching / normalization
                    # -----------------------------
                    drugs_matched, ades_matched = [], []
                    for d in entities["DRUG"]:
                        best, score = process.extractOne(d.lower(), DRUG_DICT, scorer=fuzz.token_set_ratio)
                        if score > 70:
                            drugs_matched.append(best)
                        else:
                            drugs_matched.append(d.lower())

                    for a in entities["ADE"]:
                        best, score = process.extractOne(a.lower(), ADE_DICT, scorer=fuzz.token_set_ratio)
                        if score > 70:
                            ades_matched.append(best)
                        else:
                            ades_matched.append(a.lower())

                    # Store post-processed entity dict
                    post_entities.append({"DRUG": list(set(drugs_matched)), "ADE": list(set(ades_matched))})

                    # Combined string for clustering
                    all_entities_str = " ".join(drugs_matched + ades_matched)
                    results.append(all_entities_str)
                    highlights.append(pred_labels)

                return results, highlights, post_entities

            st.info("Running BioBERT predictions + post-processing...")
            df["predicted_entities"], df["highlights"], df["post_entities"] = predict_entities(df["symptom_text"])
        else:
            st.warning("Skipping BioBERT. Using raw text.")
            df["predicted_entities"] = df["symptom_text"]
            df["highlights"] = [[(w, "TEXT") for w in t.split()] for t in df["symptom_text"]]
            df["post_entities"] = [{"DRUG": [], "ADE": t.split()} for t in df["symptom_text"]]




        # -----------------------------
        # 4️⃣ Severity Classification
        # -----------------------------
        def assign_severity(row):
            sev = str(row.get("severity", "")).lower()
            text = row["symptom_text"].lower()
            if "severe" in text or sev in ["severe", "hospitalized", "death"]:
                return "Severe"
            elif "moderate" in text or sev == "moderate":
                return "Moderate"
            else:
                return "Mild"

        df["severity_label"] = df.apply(assign_severity, axis=1)

        # -----------------------------
        # 5️⃣ Embeddings + Clustering
        # -----------------------------
        st.info("Generating embeddings for ADE/Drug clustering...")
        model_embed = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model_embed.encode(df["predicted_entities"].astype(str).tolist(), show_progress_bar=True)

        n_clusters = st.slider("Number of Clusters (K)", 2, 15, 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df["cluster"] = kmeans.fit_predict(embeddings)

        # -----------------------------
        # 6️⃣ t-SNE Visualization
        # -----------------------------
        st.info("Running t-SNE for visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        X_tsne = tsne.fit_transform(embeddings)
        df["x"] = X_tsne[:,0]
        df["y"] = X_tsne[:,1]


        # -----------------------------
        # 6. Interactive Plotly Cluster Plots
        # -----------------------------
        st.subheader("Cluster Visualizations")

        # Build hover text
        hover_text = [
            f"Entities: {ent}<br>Age Group: {age}<br>Severity: {sev}"
            for ent, age, sev in zip(df["predicted_entities"], df["age_group"], df["severity_label"])
        ]

        # Create Streamlit columns
        #col2, col3 = st.columns(2)

        # 1️⃣ Clusters by Unsupervised Cluster ID
        #with col1:
        #    fig1 = px.scatter(
        #        df, x="x", y="y", color="cluster",
        #        hover_name=hover_text,
        #        color_discrete_sequence=px.colors.qualitative.Plotly,
        #        title="Clusters"
            #)
        #    st.plotly_chart(fig1, use_container_width=True)

        # 2️⃣ Clusters by Severity
        #with col2:
        fig2 = px.scatter(
            df, x="x", y="y", color="severity_label",
            hover_name=hover_text,
            color_continuous_scale="RdYlBu",
            title="By Severity"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 3️⃣ Clusters by Age Group
        #with col3:
        fig3 = px.scatter(
            df, x="x", y="y", color="age_group",
            hover_name=hover_text,
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="By Age Group"
        )
        st.plotly_chart(fig3, use_container_width=True)


        # -----------------------------
        # 8️⃣ Token-level Highlighting
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
        # 9️⃣ Show Table + Post-Processed Entities
        # -----------------------------
        st.subheader("Clustered & Labeled Data (Post-Processed)")
        df_table = df[["predicted_entities", "post_entities", "cluster", "severity_label", "age_group"]]
        st.write(df_table.head(20))

        st.download_button(
            label="Download Clustered & Labeled Data",
            data=df_table.to_csv(index=False),
            file_name="clustered_ade_dataset_postprocessed.csv",
            mime="text/csv"
        )
else:
    st.info("Upload a CSV to start the analysis.")
