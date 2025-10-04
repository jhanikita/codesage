
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from .retrieval import retrieve_top_k


try:
    print("Loading embedding model...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading SentenceTransformer: {e}")


try:
    print("Connecting to Mistral model via Ollama...")
    llm = Ollama(model="mistral")  
    print("Mistral LLM connected successfully.")
except Exception as e:
    raise RuntimeError(f"Error initializing Ollama/Mistral model: {e}")


def generate_answer(query: str, k: int = 3) -> str:

    try:
        query_emb = embed_model.encode([query])[0]
        context_docs = retrieve_top_k(query_emb, k=k)
        print(f"Retrieved {len(context_docs)} docs for query: '{query}'")
        context_text = "\n\n".join(context_docs) if context_docs else "No relevant context found."

        prompt = (
            f"You are a helpful documentation assistant. "
            f"Use the provided context to answer clearly and concisely.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        answer = llm.invoke(prompt)  
        return answer.strip()

    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, something went wrong while generating the answer."


