import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import faiss
import numpy as np
import tempfile

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
        model_path="/Users/anishmalla/PycharmProjects/RAG-No-Embedding/mistral-7b-instruct-v0.1.Q2_K.gguf",
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=35,
        verbose=False,
    )
    output = llm(prompt, max_tokens=1024, stop=["</s>", "Question:"])
    return output['choices'][0]['text'].strip()

# --- Streamlit UI ---
st.title("PDF Q&A with Llama (Standalone)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name
    st.success("PDF uploaded successfully!")
    question = st.text_input("Ask a question about the PDF:")
    if question:
        st.write("🔄 Loading and chunking PDF...")
        chunk_size = 1000
        k = 7
        chunks = load_pdf_chunks(pdf_path, chunk_size=chunk_size)
        embeddings, embed_model = embed_chunks(chunks)
        index = build_faiss_index(embeddings)
        limited_chunks = get_top_k_chunks(question, embed_model, index, chunks, k=k)
        prompt = build_prompt(limited_chunks, question)
        answer = ask_llama(prompt)
        st.markdown(f"**Answer:** {answer}")
        st.subheader("Selected Chunks:")
        for idx, chunk, score in limited_chunks:
            st.markdown(f"**Chunk {idx}** (Similarity Score: {score:.2f}):\n{chunk}")
else:
    st.info("Please upload a PDF to get started.")

