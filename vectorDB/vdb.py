import faiss
from openai import OpenAI
import numpy as np

class FaissIndex:
    def __init__(self, dimension, api_key, index_type="flat", nlist=50, model="text-embedding-3-small"):
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.model = model
        self.index = self.create_index()
        self.client = OpenAI(api_key=api_key)
        self.stored_texts = {}
        self.current_index = 0
    
    def create_index(self):
        if self.index_type == "flat":
            index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivfflat":
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_L2)
        else:
            raise ValueError("Unsupported index type")
        
        return index
        
    def train_index(self, texts):
        embeddings = self.texts_to_embeddings(texts)
        if self.index_type == "ivfflat" and not self.index.is_trained:
            self.index.train(embeddings)
    
    def add_text(self, texts):
        embedding = self.texts_to_embeddings(texts)
        if isinstance(embedding, np.ndarray):
            self.index.add(embedding)
            for text in texts:
                self.stored_texts[self.current_index] = text
                self.current_index += 1

    def search_text(self, query_text:str, k:int=1) -> list[tuple[float,str]]:
        assert isinstance(query_text, str), "search_text: Måste vara en sträng"
        query_embedding = self.text_to_embedding(query_text)
        distances, indices = self.index.search(query_embedding, k)
        print(f"{distances} : {indices}")
        results = [(distances[0][i], self.stored_texts[indices[0][i]]) for i in range(k)]
        return results

    def text_to_embedding(self, text):
        response = self.client.embeddings.create(
            input=[text],  # Even a single text must be passed as a list
            model=self.model
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        assert embedding.shape[0] == self.dimension, f"Expected embedding dimension {self.dimension}, but got {embedding.shape[0]}"
        a = embedding.reshape(1, -1)
        #print(f"{embedding} : {a}")
        return a
    
    def texts_to_embeddings(self, texts):
        embeddings = [self.text_to_embedding(text) for text in texts]
        return np.vstack(embeddings)

