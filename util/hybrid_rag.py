import os
from llama_index import SimpleDirectoryReader, SemanticSplitterNodeParser
from llama_index import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def semantic_chunking(file_path):
    """Perform semantic chunking on the document."""
    docs = SimpleDirectoryReader(file_path).load_data()
    semantic_parser = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=Settings.embed_model
    )
    return semantic_parser.get_nodes_from_documents(docs)

def summarize_chunks_with_llm(nodes):
    """Summarize each chunk using Azure OpenAI."""
    def summarize_chunk(text):
        resp = Settings.llm.chat.completions.create(
            engine="gpt-4o",
            messages=[{"role": "user", "content": f"Summarize this excerpt in 2–3 sentences:\n\n{text}"}],
            temperature=0.0
        )
        return resp.choices[0].message.content.strip()

    return [summarize_chunk(node.text) for node in nodes]

def build_faiss_index_with_embeddings(summaries):
    """Embed summaries and build a FAISS index."""
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = embed_model.encode(summaries, convert_to_numpy=True)
    dim = embs.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embs)
    return faiss_index, embed_model

def rag_query_with_llm(query, summaries, faiss_index, embed_model, top_k=5):
    """Perform RAG query using FAISS and Azure OpenAI."""
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    _, idxs = faiss_index.search(q_emb, top_k)
    context = "\n\n".join(f"- {summaries[i]}" for i in idxs[0])
    prompt = (
        "Using these summaries as context, answer the question:\n\n"
        f"{context}\n\nQuestion: {query}"
    )
    resp = Settings.llm.chat.completions.create(
        engine="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return resp.choices[0].message.content.strip()

def hybrid_summarization_rag(file_path, query):
    """Perform hybrid summarization + RAG on a document."""
    nodes = semantic_chunking(file_path)
    summaries = summarize_chunks_with_llm(nodes)
    faiss_index, embed_model = build_faiss_index_with_embeddings(summaries)
    return rag_query_with_llm(query, summaries, faiss_index, embed_model)