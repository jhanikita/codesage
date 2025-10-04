import os
os.makedirs("./embeddings", exist_ok=True)

from sentence_transformers import SentenceTransformer
import pickle
from preprocessing import load_docs

model = SentenceTransformer('all-MiniLM-L6-v2')

docs = load_docs()
texts = [d["content"] for d in docs]
embeddings = model.encode(texts)
print(embeddings)

with open("./embeddings/faiss_index.pkl", "wb") as f:
    pickle.dump({"docs": docs, "embeddings": embeddings}, f)

print("Embeddings saved!")
