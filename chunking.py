from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import faiss
import numpy as np

# --- Step 1: Load and Chunk PDF ---
def load_pdf_chunks(pdf_path, chunk_size=1000):
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# --- Step 2: Generate Embeddings ---
def embed_chunks(chunks, model_name="all-mpnet-base-v2"):
    embed_model = SentenceTransformer(model_name)
    embeddings = embed_model.encode(chunks)
    return embeddings, embed_model

# --- Step 3: Create Vector Index with FAISS ---
def build_faiss_index(embeddings):
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

# --- Step 4: Perform Similarity Search ---
def get_top_k_chunks(query, embed_model, index, chunks, k=7):
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    # Return (index, chunk, similarity score)
    top_chunks = [(i, chunks[i], float(D[0][idx])) for idx, i in enumerate(I[0])]
    return top_chunks

# --- Step 5: Build Prompt ---
def build_prompt(indexed_chunks, query):
    prompt = (
        "You are a helpful assistant. Use the following context to answer the question as accurately as possible. "
        "If the answer is not clear from the context, say so.\n\n"
        "Context:\n"
    )
    for idx, chunk, score in indexed_chunks:
        prompt += f"[Chunk #{idx} | Similarity Score: {score:.2f}]\n{chunk.strip()}\n---\n"
    prompt += f"\nQuestion: {query}\nAnswer:"
    return prompt

# --- Step 6: Query Mistral with llama.cpp ---
def ask_llama(prompt):
    llm = Llama(
        model_path="mistral-7b-instruct-v0.1.Q2_K.gguf",
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=35,
        verbose=False,
    )
    output = llm(prompt, max_tokens=1024, stop=["</s>", "Question:"])
    return output['choices'][0]['text'].strip()

# --- Main App ---
def main():
    pdf_path = "/Users/anishmalla/PycharmProjects/RAG-No-Embedding/Atomic habits ( PDFDrive ).pdf"  # Replace with your actual file
    print("🔄 Loading and chunking PDF...")
    chunks = load_pdf_chunks(pdf_path)

    print(f"🧠 Generating embeddings for {len(chunks)} chunks...")
    embeddings, embed_model = embed_chunks(chunks)

    print("📦 Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("\n✅ RAG system ready. Ask your question!\n")

    while True:
        query = input("\n❓ Ask a question (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break

        top_chunks = get_top_k_chunks(query, embed_model, index, chunks)

        print("\n📄 Top Matching Chunks:")
        for idx, chunk, score in top_chunks:
            preview = chunk.strip().replace('\n', ' ')[:200]
            print(f"- Chunk #{idx} (Score: {score:.2f}): {preview}...")

        prompt = build_prompt(top_chunks, query)
        print("\n🤖 Querying Mistral...")
        answer = ask_llama(prompt)

        print("\n💡 Answer:\n", answer)

if __name__ == "__main__":
    main()
