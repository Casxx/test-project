from util.hybrid_rag import hybrid_summarization_rag

def qa_feature(file_path, query):
    """Perform Q&A using hybrid summarization + RAG."""
    return hybrid_summarization_rag(file_path, query)