#this basically finds similarity between query and docs present using vector embedding
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

with open("./embeddings/faiss_index.pkl", "rb") as f:
    data = pickle.load(f)

docs = data["docs"]
embeddings = data["embeddings"]

def retrieve_top_k(query_embedding, k=5):
    sims = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(sims)[-k:][::-1]
    top_docs = [docs[i]["content"] for i in top_indices]
    return top_docs
