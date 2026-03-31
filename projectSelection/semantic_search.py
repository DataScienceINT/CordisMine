import faiss
import spacy
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import csv
from model2vec import StaticModel

# --- CONFIG ---
SIMILARITY_THRESHOLD = 0.5
INDEX_PATH = 'sentence_index_bge.faiss'
METADATA_PATH = 'sentence_metadata_bge.npy'
CSV_OUTPUT = 'query_matches_bge.csv'
SENTENCE_CSV = 'split_sentences.csv'

# --- LOAD TOOLS ---
#model = StaticModel.from_pretrained("m2v_model_voc")
base_model_name = "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(base_model_name)
nlp = spacy.load("en_core_sci_sm", disable=["ner"])

def split_into_sentences(documents):
    sentence_data = []  
    texts = [d["objective"] for d in documents]
    doc_map = {d["objective"]: (d["id"], d["scores"]) for d in documents}
    
    for doc in nlp.pipe(texts, batch_size=32):
        orig_text = doc.text
        doc_id, doc_score = doc_map[orig_text]
        for sent in doc.sents:
            sentence = sent.text.strip()
            if sentence:
                sentence_data.append((doc_id, sentence, doc_score))
    # Save sentences to CSV
    df_sentences = pd.DataFrame(sentence_data, columns=["id", "sentence", "original_score"])
    df_sentences.to_csv(SENTENCE_CSV, index=False)
    print(f"Saved split sentences to {SENTENCE_CSV}")
    return sentence_data

def build_index(sentences):
    texts = [s[1] for s in sentences]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    np.save(METADATA_PATH, np.array(sentences, dtype=object))

    return index, sentences, embeddings

def search_index(index, sentences, queries, threshold=0.7):
    query_embs = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    results = []
    for query, q_emb in zip(queries, query_embs):
        D, I = index.search(q_emb.reshape(1, -1), k=len(sentences))
        for j, i in enumerate(I[0]):
            sim_score = float(D[0][j])
            if sim_score >= threshold:
                doc_id, sentence, orig_score = sentences[i]
                results.append({
                    "query": query,
                    "id": doc_id,
                    "text": sentence,
                    "similarity_score": round(sim_score, 4),
                    "classification_score": orig_score
                })
    return results


queries = [
    # Non animal testing and alternatives
    "non-animal methods for toxicity testing",
    "non-animal methods for drug development",
    "non-animal methods for disease modelling",
    "new approach methods for toxicity testing",
    "new approach methods for drug development",
    "new approach methods for disease modelling",
    "next generation risk assessment",
    "models to reduce animal testing",
    "reduction of animal testing in research",
    "replacement of animal testing",
    "reducing animal use in scientific research",
    "alternatives to animal experiments",
    "alternatives to animal experiments for toxicity testing",
    "alternatives to animal experiments for safety assessment",
    "alternatives to animal experiments for disease modelling",
    "developing non-animal methods for toxicity testing",
    "developing non-animal methods for safety assessment",
    "developing non-animal methods for disease modelling",
    "developing organ-on-a-chip technologies",
    "advancing alternatives to animal testing",

    # In silico and computational methods
    "in silico methods",
    "PBPK modelling",
    "PBTK modelling",
    "physiologically based kinetic modeling",
    "QSAR modelling",
    "structure activity relationships",

    # in vitro
    "microfluidics for toxicity testing",
    "microfluidics for drug development",
    "microfluidics for disease modelling",
    "on-chip for toxicity testing",
    "on-chip for drug testing",
    "on-chip for disease modelling",
    "organoids for toxicity testing",
    "organoids for drug development",
    "organoids for disease modelling",
    "iPSC for toxicity testing",
    "iPSC for drug development",
    "iPSC for disease modelling",
    "human stem cell disease models",
    "bioprinting for toxicity testing",
    "bioprinting for drug development",
    "bioprinting for disease modelling",
    "high-throughput testing for drug development",
    "high-throughput screening for toxicity testing",
    "high-throughput screening for chemicals",
    "in vitro assays replacing animal models",
    "human-relevant in vitro models",

    # refinement and reduction
    "laboratory animal welfare",
    "refinement of animal testing",
    "reducing animal use in scientific research",
    "3Rs",

    # small animals
    "zebrafish embryo",
    "drosophila",
    "C.elegans"

]

with open('all_with_predictions_13062025.csv') as f:
    projects = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]
    
sentences = split_into_sentences(projects)
index, metadata, _ = build_index(sentences)
matches = search_index(index, metadata, queries, SIMILARITY_THRESHOLD)

df = pd.DataFrame(matches)
df.to_csv(CSV_OUTPUT, index=False)

print(f"Saved {len(df)} matches to {CSV_OUTPUT}")
