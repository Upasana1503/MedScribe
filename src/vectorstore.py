import json
import os
from pathlib import Path
from typing import Any, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata: List[Any] = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_texts(self, texts: List[str], metadatas: List[Any] | None = None):
        if not texts:
            raise ValueError("No texts were provided to build the vector store.")
        print(f"[INFO] Building vector store from {len(texts)} text segments...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        if metadatas is None:
            metadatas = [{"text": text} for text in texts]
        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        embeddings = _normalize(embeddings)
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index (cosine similarity).")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.json")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.json")
        pkl_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        if Path(meta_path).is_file():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        elif Path(pkl_path).is_file():
            import pickle
            with open(pkl_path, "rb") as f:
                self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        query_embedding = _normalize(query_embedding)
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": int(idx), "score": float(score), "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5, log: bool = True):
        if log:
            print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype("float32")
        return self.search(query_emb, top_k=top_k)

