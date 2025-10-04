# CodeSage 

**Your magical assistant for developer documentation**

CodeSage leverages RAG (Retrieval-Augmented Generation) and LLM technology to help developers query their documentation and get precise, context-aware answers. Upload your docs and embeddings, and CodeSage becomes your personal copilot for faster coding and learning.

---

## Features

- Chat-style interface using **Streamlit**
- Smart document retrieval with **FAISS**
- Answers generated via **Mistral**
- Handles Markdown and text documentation
- Easy deployment with **Docker**

---

## setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/codesage.git
cd codesage

### 2. Create virtual environment

python3 -m venv .venv
source .venv/bin/activate

### 3. Install dependencies

pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit uvicorn

### 4. Run backend

uvicorn src.api:app --host 0.0.0.0 --port 8000

### 5. Run frontend (in another terminal)

streamlit run src/frontend.py --server.port 8501 --server.headless true


### Note - 
1. Make sure FAISS index exists: embeddings/faiss_index.pkl 
2. For Ollama Mistral, pull model locally using - ollama pull mistral
3. You can also add more documentation in the data/ folder to make the assistant smarter and more accurate.
4. You can use docker as well 