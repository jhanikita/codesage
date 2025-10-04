#preprocessing script - it cleans, divide docs into chunks, 
import os
import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip()

def chunk_text_with_overlap(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  
    return chunks

def load_docs(folder_path="./data"):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".md") or file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                content = clean_text(f.read())
                chunks = chunk_text_with_overlap(content)
                for c in chunks:
                    docs.append({"title": file, "content": c})
    return docs

if __name__ == "__main__":
    docs = load_docs()
    print(f"Loaded {len(docs)} chunks")

