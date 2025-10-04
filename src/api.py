import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from .retrieval import retrieve_top_k


try:
    from langchain_ollama import OllamaLLM
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False


try:
    from langchain_community.llms import LlamaCpp
    LOCAL_MODEL_AVAILABLE = True
except ImportError:
    LOCAL_MODEL_AVAILABLE = False


embed_model = SentenceTransformer('all-MiniLM-L6-v2')


if MISTRAL_AVAILABLE:
    try:
        llm = OllamaLLM(
            model="mistral",
            temperature=0.2,
            top_p=0.9,
            max_tokens=300
        )
        print("model working")
    except Exception:
        MISTRAL_AVAILABLE = False

elif not MISTRAL_AVAILABLE:
    raise RuntimeError("No LLM available. Install ollama ")


app = FastAPI(title="CodeSage")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3  

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Welcome to CodeSage! Use /ask to query."}

@app.post("/ask")
def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        query_emb = embed_model.encode([request.query])[0]
        context_docs = retrieve_top_k(query_emb, k=request.top_k)
        context_text = "\n\n".join(context_docs) if context_docs else "No relevant context found."

        prompt = f"Context:\n{context_text}\n\nQuestion: {request.query}\nAnswer:"

        if MISTRAL_AVAILABLE:
            answer = llm.invoke(prompt)
        else:
            answer = llm(prompt, max_tokens=300)

        return {"query": request.query, "answer": answer.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")
