import os
from sentence_transformers import SentenceTransformer
import faiss
from config.params import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, FAISS_INDEX_PATH, TOP_K
from API.api_call import runner
from util.utils import extract_text_from_pdf, extract_text_from_word, extract_text_from_image

def chunk_text(text, chunk_size, overlap):
    """Chunk text into overlapping windows."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def summarize_chunks(chunks):
    """Summarize each chunk using the runner function."""
    return [runner(f"Summarize this: {chunk}") for chunk in chunks]

def embed_summaries(summaries):
    """Embed summaries using SentenceTransformer."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(summaries)

def build_faiss_index(embeddings):
    """Build a FAISS index for the embeddings."""
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

def query_faiss_index(query, index_path, top_k):
    """Query the FAISS index for top-k results."""
    index = faiss.read_index(index_path)
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return indices

def hybrid_summarization_rag(file_path, query):
    """Perform hybrid summarization + RAG on a document."""
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_word(file_path)
    elif file_path.endswith(('.png', '.jpg', '.jpeg')):
        text = extract_text_from_image(file_path)
    else:
        raise ValueError("Unsupported file type")

    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    summaries = summarize_chunks(chunks)
    embeddings = embed_summaries(summaries)
    build_faiss_index(embeddings)

    top_indices = query_faiss_index(query, FAISS_INDEX_PATH, TOP_K)
    top_summaries = [summaries[i] for i in top_indices[0]]

    final_answer = runner(f"Answer this based on the following summaries: {top_summaries}. Query: {query}")
    return final_answer