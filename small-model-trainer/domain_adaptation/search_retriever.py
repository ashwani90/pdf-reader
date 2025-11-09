# ============================
# search_retriever.py
# ============================
"""
This module provides a simple semantic search retriever using FAISS.
It reads your text corpus, builds embeddings, and lets you perform
semantic similarity search queries.
"""

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse
import pickle


def build_faiss_index(text_file, index_path="faiss_index", chunk_size=500, chunk_overlap=50):
    """
    Builds a FAISS index from the given text file.

    Parameters:
        text_file (str): Path to a .txt file containing raw text (your news corpus)
        index_path (str): Directory where the FAISS index and metadata will be saved
        chunk_size (int): Max number of characters per text chunk
        chunk_overlap (int): Overlap between consecutive chunks
    """
    print(f"📖 Reading corpus from {text_file}...")
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    # ------------------------------
    # Split text into chunks
    # ------------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    print(f"✅ Split corpus into {len(chunks)} chunks.")

    # ------------------------------
    # Load embedding model
    # ------------------------------
    # You can replace this with any SentenceTransformer model
    # For small GPUs/CPUs, MiniLM-L6-v2 is very efficient
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Compute embeddings for each chunk
    print("🔹 Encoding chunks into embeddings...")
    embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    # ------------------------------
    # Create FAISS index
    # ------------------------------
    dim = embeddings.shape[1]  # embedding dimension
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"✅ FAISS index built with {index.ntotal} vectors.")

    # Save index and metadata
    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, os.path.join(index_path, "corpus.index"))

    with open(os.path.join(index_path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"💾 Saved FAISS index and chunks to {index_path}/")


def search_faiss(query, index_path="faiss_index", top_k=3):
    """
    Performs semantic search over the FAISS index.

    Parameters:
        query (str): Search query
        index_path (str): Path where the FAISS index and chunks are stored
        top_k (int): Number of top results to return
    """
    print("🔍 Loading index and embeddings...")
    index = faiss.read_index(os.path.join(index_path, "corpus.index"))
    with open(os.path.join(index_path, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

    # Load same embedder used for building index
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = embedder.encode([query], convert_to_numpy=True)

    # Search for top-k nearest chunks
    D, I = index.search(np.array(query_vec), top_k)
    results = [(chunks[i], float(D[0][idx])) for idx, i in enumerate(I[0])]

    print(f"\n🔹 Top {top_k} results for query: '{query}'\n")
    for rank, (text, score) in enumerate(results, 1):
        print(f"{rank}. (distance={score:.4f})\n{text[:400]}...\n")

    return [r[0] for r in results]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or query a FAISS retriever.")
    parser.add_argument("--build", action="store_true", help="Build a FAISS index from a text file.")
    parser.add_argument("--query", type=str, help="Run a search query.")
    parser.add_argument("--text_file", type=str, help="Path to text file (for --build).")
    parser.add_argument("--index_path", default="faiss_index", help="Where to save/load the FAISS index.")
    args = parser.parse_args()

    if args.build:
        if not args.text_file:
            print("❌ Please specify --text_file to build the index.")
        else:
            build_faiss_index(args.text_file, args.index_path)

    elif args.query:
        search_faiss(args.query, args.index_path)

    else:
        print("❌ Please specify either --build or --query.")
