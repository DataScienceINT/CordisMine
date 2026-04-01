import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# 1. Load your cleaned projects file
df = pd.read_csv("CORDIS_NAM_PROJECTS.csv")

# 2. Build a text field for retrieval
texts = (
    df["acronym"].fillna("") + " - " +
    df["title"].fillna("")   + " " +
    df["objective"].fillna("")
).tolist()

# 3. BM25 index
tokenized = [t.lower().split() for t in texts]
bm25 = BM25Okapi(tokenized)

# 4. Embeddings (local, free)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# 5. Save everything you need
np.save("project_embeddings.npy", embeddings)