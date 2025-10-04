# RAG Pipeline: Theory and Code Implementation

## What is RAG?

RAG is a technique that combines the power of large language models (LLMs) with external knowledge retrieval. Instead of relying solely on the model's training data, RAG allows the AI to access and use specific, up-to-date information from external sources to generate more accurate and contextual responses.

## How RAG Works

The process follows these key steps:

### 1. Knowledge Preparation

Your documents, PDFs, web pages, or other text sources are broken down into smaller chunks (usually 200-1000 words). These chunks are then converted into numerical representations called embeddings.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import openai

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Split documents into chunks
def prepare_documents(documents):
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_text(doc.content)
        for chunk in doc_chunks:
            chunks.append({
                'text': chunk,
                'metadata': doc.metadata
            })
    return chunks

# Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
chunks = prepare_documents(documents)
chunk_embeddings = embeddings.embed_documents([c['text'] for c in chunks])
```

### 2. Storage and Indexing

The embeddings are stored in a vector database where they can be quickly searched based on semantic similarity.

```python
import faiss
import numpy as np
from typing import List, Dict

class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.chunks = []
    
    def add_documents(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Add document chunks and their embeddings to the store"""
        embeddings_array = np.array(embeddings).astype('float32')
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        self.index.add(embeddings_array)
        self.chunks.extend(chunks)
    
    def search(self, query_embedding: List[float], k: int = 5):
        """Search for most similar chunks"""
        query_array = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_array)
        
        scores, indices = self.index.search(query_array, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                results.append({
                    'chunk': self.chunks[idx],
                    'score': float(score),
                    'rank': i
                })
        return results

# Initialize vector store
vector_store = VectorStore(dimension=1536)  # OpenAI ada-002 dimension
vector_store.add_documents(chunks, chunk_embeddings)
```

### 3. Query Processing

When a user asks a question, that query is also converted into an embedding using the same model.

```python
def process_query(query: str, embeddings_model):
    """Convert user query to embedding"""
    query_embedding = embeddings_model.embed_query(query)
    return query_embedding

# Example usage
user_query = "What is machine learning?"
query_embedding = process_query(user_query, embeddings)
```

### 4. Retrieval

The system searches the vector database to find the most semantically similar document chunks.

```python
def retrieve_relevant_context(query: str, vector_store: VectorStore, 
                            embeddings_model, k: int = 5):
    """Retrieve relevant chunks for the query"""
    # Get query embedding
    query_embedding = embeddings_model.embed_query(query)
    
    # Search for similar chunks
    results = vector_store.search(query_embedding, k=k)
    
    # Format context
    context_chunks = []
    for result in results:
        context_chunks.append({
            'text': result['chunk']['text'],
            'metadata': result['chunk']['metadata'],
            'relevance_score': result['score']
        })
    
    return context_chunks

# Retrieve context
relevant_context = retrieve_relevant_context(user_query, vector_store, embeddings)
```

### 5. Generation

The retrieved context is combined with the original query and fed to an LLM.

```python
import openai
from typing import List

def generate_response(query: str, context_chunks: List[Dict], 
                     model: str = "gpt-4") -> str:
    """Generate response using retrieved context"""
    
    # Prepare context string
    context_text = "\n\n".join([
        f"Source {i+1}: {chunk['text']}" 
        for i, chunk in enumerate(context_chunks)
    ])
    
    # Create prompt
    system_prompt = """You are a helpful assistant. Use the provided context to answer the user's question. 
    If the context doesn't contain relevant information, say so clearly."""
    
    user_prompt = f"""Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above."""
    
    # Generate response
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    
    return response.choices[0].message.content

# Generate final response
answer = generate_response(user_query, relevant_context)
```

## Complete RAG Pipeline

Here's how all components work together:

```python
class RAGPipeline:
    def __init__(self, embeddings_model, llm_model="gpt-4"):
        self.embeddings_model = embeddings_model
        self.llm_model = llm_model
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def ingest_documents(self, documents: List[str]):
        """Process and store documents in the vector database"""
        print("📄 Processing documents...")
        
        # Split documents
        all_chunks = []
        for i, doc in enumerate(documents):
            chunks = self.text_splitter.split_text(doc)
            for chunk in chunks:
                all_chunks.append({
                    'text': chunk,
                    'metadata': {'doc_id': i, 'source': f'document_{i}'}
                })
        
        # Create embeddings
        print("🔢 Creating embeddings...")
        chunk_embeddings = self.embeddings_model.embed_documents(
            [chunk['text'] for chunk in all_chunks]
        )
        
        # Store in vector database
        print("💾 Storing in vector database...")
        self.vector_store = VectorStore(dimension=len(chunk_embeddings[0]))
        self.vector_store.add_documents(all_chunks, chunk_embeddings)
        
        print(f"✅ Processed {len(all_chunks)} chunks from {len(documents)} documents")
    
    def query(self, question: str, k: int = 5) -> Dict:
        """Query the RAG pipeline"""
        if not self.vector_store:
            return {"error": "No documents ingested yet"}
        
        print(f"🔍 Searching for: '{question}'")
        
        # Retrieve relevant context
        relevant_context = retrieve_relevant_context(
            question, self.vector_store, self.embeddings_model, k=k
        )
        
        # Generate response
        print("🤖 Generating response...")
        answer = generate_response(question, relevant_context, self.llm_model)
        
        return {
            'question': question,
            'answer': answer,
            'sources': relevant_context,
            'num_sources_used': len(relevant_context)
        }

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    rag_pipeline = RAGPipeline(embeddings)
    
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data...",
        "Deep learning uses neural networks with multiple layers to process data and make predictions...",
        "Natural language processing enables computers to understand and generate human language..."
    ]
    
    # Ingest documents
    rag_pipeline.ingest_documents(documents)
    
    # Query the pipeline
    result = rag_pipeline.query("What is machine learning?")
    
    print(f"\n📝 Question: {result['question']}")
    print(f"🎯 Answer: {result['answer']}")
    print(f"📚 Used {result['num_sources_used']} sources")
```

## Core Components

**Document Store**: Where your processed documents and their embeddings live (vector databases, traditional databases, or hybrid systems)

**Retriever**: The component that finds relevant information based on the query (dense retrievers using embeddings, sparse retrievers using keywords, or hybrid approaches)

**Generator**: The LLM that creates the final response using both the query and retrieved context

**Pipeline Logic**: The orchestration layer that manages the flow between components

## Advanced Techniques

### Hybrid Search (Dense + Sparse)

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, vector_store, chunks):
        self.vector_store = vector_store
        self.chunks = chunks
        
        # Initialize BM25 for sparse retrieval
        tokenized_chunks = [chunk['text'].split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
    
    def hybrid_search(self, query, k=5, alpha=0.5):
        """Combine dense and sparse retrieval"""
        # Dense retrieval (semantic)
        query_embedding = embeddings.embed_query(query)
        dense_results = self.vector_store.search(query_embedding, k=k*2)
        
        # Sparse retrieval (keyword)
        tokenized_query = query.split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        
        # Combine scores
        combined_results = []
        for i, chunk in enumerate(self.chunks):
            dense_score = 0
            for result in dense_results:
                if result['chunk'] == chunk:
                    dense_score = result['score']
                    break
            
            sparse_score = sparse_scores[i]
            combined_score = alpha * dense_score + (1 - alpha) * sparse_score
            
            combined_results.append({
                'chunk': chunk,
                'score': combined_score,
                'dense_score': dense_score,
                'sparse_score': sparse_score
            })
        
        # Return top k results
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        return combined_results[:k]
```

### Query Expansion

```python
def expand_query(query: str, llm_model="gpt-3.5-turbo") -> List[str]:
    """Generate additional related queries"""
    prompt = f"""Given this query: "{query}"
    
    Generate 3 related questions that might help find relevant information:
    1.
    2. 
    3."""
    
    response = openai.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    # Parse the response to extract queries
    expanded_queries = [query]  # Include original
    lines = response.choices[0].message.content.strip().split('\n')
    for line in lines:
        if line.strip() and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
            expanded_queries.append(line[2:].strip())
    
    return expanded_queries
```

## Why RAG is Valuable

**Accuracy**: Reduces hallucinations by grounding responses in real data
**Timeliness**: Can work with constantly updated information without retraining models
**Cost-Effective**: Avoids the expense of fine-tuning large models
**Transparency**: You can see exactly which sources informed each response
**Domain-Specific**: Works excellently with specialized knowledge bases

## Common Challenges and Solutions

**Chunking Strategy**: Finding the right balance between chunk size and context preservation. Too small loses context, too large dilutes relevance.

**Retrieval Quality**: Poor retrieval leads to irrelevant context. Solutions include hybrid search (combining dense and sparse retrieval), query expansion, and re-ranking retrieved results.

**Context Window Management**: LLMs have token limits, so you need strategies for selecting and prioritizing the most relevant retrieved information.

RAG has become essential for building AI applications that need to work with specific, current, or proprietary information while maintaining the conversational abilities of modern language models.