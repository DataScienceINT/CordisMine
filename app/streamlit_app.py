import streamlit as st

import pandas as pd
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pathlib import Path
import altair as alt

st.set_page_config(page_title="🔬EU NAMs Dashboard", layout="wide")

# -----------------------------
# Load data + build indexes
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_index():
    df = pd.read_csv(BASE_DIR / "CORDIS_NAM_PROJECTS.csv")
    df = df.where(pd.notnull(df), None)

    texts = (
        df["acronym"].fillna("") + " - " +
        df["title"].fillna("")   + " " +
        df["objective"].fillna("")
    ).tolist()

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    embeddings = np.load(BASE_DIR / "project_embeddings.npy")

    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    return df, texts, tokenized, bm25, embeddings, emb_model


df, texts, tokenized, bm25, embeddings, emb_model = load_index()

# -----------------------------
# App UI
# -----------------------------
st.title("🐾EU-Funded New Approach Methodologies/3Rs Dashboard")
st.write("This dashboard helps to explore NAMs and 3Rs-related EU projects from " \
"FP7, Horizon 2020 and Horizon EU.")

st.subheader("🎯Project Dataset")
st.write("Full downloadable list of projects considered relevant to NAMs and 3Rs. " \
         "Double-click to unroll descriptions.")
st.dataframe(df, use_container_width=True)

# ================================
# VISUALISATION SECTION (FULL FIX)
# ================================

st.header("📈Project Label Visualisations")
st.write("Categories and labels were assigned using a semi-automated approach (see paper). " \
"Categories included toxicology terms, organ types, disease types, lab methods, in silico methods, small animal models and key policy terms. " \
"Hover over bars for details. ")
# Load the file
labels_df = pd.read_csv(BASE_DIR / "ALL_PROJECT_LABELS.csv")

# --------------------------
# HARD CLEANING FUNCTION
# --------------------------
def normalize(x):
    if pd.isna(x):
        return "Unknown"
    x = str(x)
    # Remove common hidden characters
    for bad in ["\xa0", "\u200b", "\ufeff", "\t"]:
        x = x.replace(bad, " ")
    x = x.strip()
    if x == "":
        return "Unknown"
    return x

# Apply cleaning
labels_df["Category"] = labels_df["Category"].apply(normalize)
labels_df["Label"] = labels_df["Label"].apply(normalize)

# Normalize to consistent case
labels_df["Category"] = labels_df["Category"].str.title()
labels_df["Label"] = labels_df["Label"].str.title()

# Drop completely empty rows (if any)
labels_df = labels_df.dropna(subset=["Category", "Label"], how="all")

# --------------------------
# CATEGORY/LABEL DISTRIBUTION
# --------------------------
# Create counts dataframe 
cat_counts = labels_df["Category"].value_counts().reset_index() 
# Rename columns 
cat_counts.columns = ["Category", "Count"] 
# Ensure numeric AFTER rename 
cat_counts["Count"] = pd.to_numeric(cat_counts["Count"], errors="coerce") 

cat_chart = (
    alt.Chart(cat_counts)
    .mark_bar()
    .encode(
        x=alt.X("Category:N", sort='-y', axis=alt.Axis(labelAngle=-45)),
        y="Count:Q",
        color=alt.Color("Category:N", legend=None),
        tooltip=["Category", "Count"]
    )
    .properties(title="Frequency of Categories", width="container")
)

st.altair_chart(cat_chart, use_container_width=True)

# --------------------------
# LABEL DISTRIBUTION (WITH CATEGORY LEGEND)
# --------------------------

# Count labels + keep their corresponding category
label_counts = (
    labels_df.groupby(["Label", "Category"])
    .size()
    .reset_index(name="Count")
)

# Sort by count desc
label_counts = label_counts.sort_values("Count", ascending=False)

label_chart = (
    alt.Chart(label_counts.head(30))
    .mark_bar()
    .encode(
        x=alt.X("Label:N", sort='-y',
                axis=alt.Axis(labelAngle=-75, labelLimit=300)),
        y="Count:Q",
        color=alt.Color("Category:N", legend=alt.Legend(title="Category")),
        tooltip=["Label", "Category", "Count"]
    )
    .properties(title="Most Common Labels (Colored by Category)",
                width="container", height=500)
)

st.altair_chart(label_chart, use_container_width=True)

# -----------------------------
# TRACEABILITY SECTION
# -----------------------------
st.header("💡 Explore Project Labels and Categories in Detail")

# Sidebar filters
selected_category = st.selectbox("Filter by Category", ["All"] + sorted(labels_df["Category"].unique()))
selected_label = st.selectbox("Filter by Label", ["All"] + sorted(labels_df["Label"].unique()))

# Merge df with label data so we can filter
merged = labels_df.copy()

# Apply filtering
filtered = merged.copy()
if selected_category != "All":
    filtered = filtered[filtered["Category"] == selected_category]

if selected_label != "All":
    filtered = filtered[filtered["Label"] == selected_label]

st.write(f"### Results: {len(filtered)} label(s)")
st.dataframe(filtered[["acronym", "title", "Category", "Label", "objective"]], use_container_width=True)

# -----------------------------
# Hybrid search
# -----------------------------
st.header("🔎 Search query across project dataset")
query = st.text_input("Enter a query (e.g. 'neurodegeneration', 'organ-on-chip', 'QSAR toxicity'):")

def hybrid_search(query, alpha=0.5, threshold=0.25):
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)

    q_emb = emb_model.encode([query], convert_to_numpy=True)[0]
    norm_docs = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    norm_q = q_emb / (np.linalg.norm(q_emb) + 1e-8)
    cos_scores = norm_docs @ norm_q

    # Normalize both score types
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    bm25_n = norm(bm25_scores)
    cos_n = norm(cos_scores)

    # Hybrid score
    hybrid = alpha * bm25_n + (1 - alpha) * cos_n

    # Filter by threshold
    idx = np.where(hybrid >= threshold)[0]
    idx = idx[np.argsort(-hybrid[idx])]  # sort descending

    # Build results dataframe including scores
    results = df.iloc[idx].copy()
    results["similarity_score"] = hybrid[idx].round(3)

    return results

# Only search if a query exists
top_results = None
if query:
    results = hybrid_search(query)

    top_results = results

    if len(results) == 0:
        st.write("No results above threshold.")
    else:
        st.write(f"### 🔎 Top {len(results)} matching projects")
        st.dataframe(results, use_container_width=True)

# -----------------------------
# Prepare JSON for WebLLM
# -----------------------------
def compress_row(row):
    return {
        "acronym": row.get("acronym", "") or "",
        "title": row.get("title", "") or "",
        "objective": (row.get("objective", "") or "")[:400],

    }

if top_results is None or top_results.empty:
    compressed_rows = []
else:
    # ✔ limit what the LLM sees to 8 rows
    compressed_rows = [compress_row(r) for _, r in top_results.head(8).iterrows()]

dataset_json = json.dumps(compressed_rows)

# -----------------------------
# WebLLM HTML + JS
# -----------------------------
html_template = r"""
<div style="border:1px solid #444; padding:10px; border-radius:8px; margin-top:1rem;">

  <h3 style="
      margin-top:0;
      color:#990a00;
      font-family:'Source Sans Pro', sans-serif;
      font-size:22px;
      font-weight:600;">
      💬 LLM short synthesis of the top 8 projects matching your query
  </h3>

  <p style="
      margin-top:4px;
      color:#cccccc;
      font-family:'Source Sans Pro', sans-serif;
      font-size:14px;
      font-style:italic;">
      Answers generated by LLMs may contain mistakes.
  </p>

  <textarea id="chat" style="
      width:100%;
      height:320px;
      background:#111;
      color:#0f0;
      font-family:'Source Code Pro', monospace;
      font-size:14px;
  "></textarea>

</div>

<script id="project-data" type="application/json">__DATASET_JSON__</script>
<script id="user-query" type="application/json">__USER_QUERY__</script>

<script type="module">
  import * as webllm from "https://esm.run/@mlc-ai/web-llm";

  async function main() {
    const chatBox = document.getElementById("chat");
    const rows = JSON.parse(document.getElementById("project-data").textContent || "[]");
    const userQuestion = JSON.parse(document.getElementById("user-query").textContent || "null");

    if (!userQuestion || rows.length === 0) {
      chatBox.value = "Enter a question above. The LLM will use matched projects as context.";
      return;
    }

    chatBox.value += "Loading model…\n";

    const engine = await webllm.CreateMLCEngine(
      "Llama-3.2-1B-Instruct-q4f16_1-MLC",
      {
        initProgressCallback: (p) => {
          const pct = Math.round((p.progress || 0) * 100);
          chatBox.value += `[loading] ${pct}% - ${p.text || ""}\n`;
        },
        safetyCheck: "none"
      }
    );

    chatBox.value += "\nModel ready. Generating answer…\n\n";

    const context = rows.map(r => `
Acronym: ${r.acronym}
Title: ${r.title}
Objective: ${r.objective}
Start: ${r.startDate}, End: ${r.endDate}
Funding: ${r.ecMaxContribution}
---`).join("\n");

    const prompt = `
You are helping a researcher understand a set of EU-funded scientific projects.

Below are several short snippets of project descriptions.
Please explain the scientific themes and connect them to the user's question.

Project snippets:
${context}

User question: ${userQuestion}

Provide:
1. Main themes
2. Purpose of the projects
3. Connection to the user's question
`;

    const reply = await engine.chat.completions.create({
      messages: [{ role: "user", content: prompt }],
      temperature: 0.2,
    });

    chatBox.value += "Assistant: " +
      (reply.choices?.[0]?.message?.content || "No answer.") + "\n";
  }

  main();
</script>
"""

safe_html = (
    html_template
    .replace("__DATASET_JSON__", dataset_json)
    .replace("__USER_QUERY__", json.dumps(query or ""))
)

st.components.v1.html(safe_html, height=550)
