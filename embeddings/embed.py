from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load the model locally. sentence-transformers is pre-installed.
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> List[float]:
        emb = self.model.encode(text)
        return emb.tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embs = self.model.encode(texts)
        return embs.tolist()
