import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any
from embeddings.embed import EmbeddingGenerator

class FAISSDatabase:
    def __init__(self, data_dir: str = None, embedder = None):
        if data_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(project_root, "data")
        else:
            self.data_dir = data_dir
        
        self.index_path = os.path.join(self.data_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(self.data_dir, "faiss_metadata.json")
        self.embedder = embedder or EmbeddingGenerator()
        
        # Determine embedding dimension by encoding a dummy string
        dummy_emb = self.embedder.get_embedding("dummy")
        self.dimension = len(dummy_emb)
        
        self.index = None
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        
        self.load()

    def load(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.texts = data.get("texts", [])
                    self.metadatas = data.get("metadatas", [])
            except Exception as e:
                self._init_empty_index()
        else:
            self._init_empty_index()

    def _init_empty_index(self):
        # We use IndexFlatIP for Cosine Similarity (requires normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.texts = []
        self.metadatas = []

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump({"texts": self.texts, "metadatas": self.metadatas}, f, indent=4)

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        if not texts:
            return
        embeddings = self.embedder.get_embeddings(texts)
        emb_arr = np.array(embeddings, dtype=np.float32)
        # Normalize vectors for cosine similarity (Inner Product index)
        norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
        emb_arr = emb_arr / (norms + 1e-8)
        
        self.index.add(emb_arr)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.save()

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        
        q_emb = np.array([self.embedder.get_embedding(query)], dtype=np.float32)
        # Normalize query vector
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        
        actual_k = min(k, self.index.ntotal)
        distances, indices = self.index.search(q_emb, actual_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.texts):
                continue
            res = self.metadatas[idx].copy()
            res["text"] = self.texts[idx]
            res["score"] = float(distances[0][i])
            results.append(res)
        return results

    def clear(self):
        self._init_empty_index()
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
            except:
                pass
        if os.path.exists(self.metadata_path):
            try:
                os.remove(self.metadata_path)
            except:
                pass
