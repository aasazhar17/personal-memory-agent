import os
import time
from typing import List, Any

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        
        # If we are not on Render, attempt local loading
        is_render = os.environ.get("RENDER") is not None
        if not is_render:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
            except ImportError:
                print("sentence-transformers not installed. Falling back to Hugging Face Inference API.")

    def get_embedding(self, text: str) -> List[float]:
        if self.model:
            emb = self.model.encode(text)
            return emb.tolist()
        else:
            res = self._query_hf_api(text)
            return self._clean_embedding(res)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.model:
            embs = self.model.encode(texts)
            return embs.tolist()
        else:
            res = self._query_hf_api(texts)
            return self._clean_embeddings(res, len(texts))

    def _query_hf_api(self, inputs: Any) -> Any:
        import requests
        token = os.environ.get("HF_TOKEN")
        # Format the model path: sentence-transformers/all-MiniLM-L6-v2
        model_id = f"sentence-transformers/{self.model_name}"
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
            
        payload = {"inputs": inputs}
        
        for attempt in range(3):
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=15)
                # Handle model loading / service unavailable
                if response.status_code == 503:
                    time.sleep(4)
                    continue
                
                response.raise_for_status()
                res = response.json()
                
                if isinstance(res, dict) and "error" in res:
                    err_msg = res.get("error", "")
                    if "loading" in err_msg.lower():
                        time.sleep(4)
                        continue
                    raise Exception(err_msg)
                return res
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(4)
        raise RuntimeError("Failed to query Hugging Face Inference API after 3 attempts.")

    def _clean_embedding(self, res: Any) -> List[float]:
        if not isinstance(res, list):
            raise ValueError(f"Expected list response from HF API, got {type(res)}")
        # Drill down to find the first list of numbers
        while len(res) > 0 and isinstance(res[0], list):
            res = res[0]
        return [float(x) for x in res]

    def _clean_embeddings(self, res: Any, expected_count: int) -> List[List[float]]:
        if not isinstance(res, list):
            raise ValueError(f"Expected list response from HF API, got {type(res)}")
        
        # Recursively search for all lists of numbers (vectors)
        def find_vectors(item):
            if not isinstance(item, list):
                return []
            if len(item) == 0:
                return []
            if isinstance(item[0], (int, float)) and not isinstance(item[0], list):
                return [item]
            vectors = []
            for child in item:
                vectors.extend(find_vectors(child))
            return vectors
            
        vecs = find_vectors(res)
        if len(vecs) != expected_count:
            raise ValueError(f"Expected {expected_count} embeddings, but found {len(vecs)}")
        return [[float(x) for x in v] for v in vecs]
