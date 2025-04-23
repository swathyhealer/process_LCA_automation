from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

class GteLargeEmbModel:
    def __init__(self):
        self.model=SentenceTransformer('thenlper/gte-large')
    def get_embeddings(self,sentences):
        embeddings = self.model.encode(sentences,convert_to_numpy=True).tolist()
        return embeddings
    def get_cosine_similarity(emb1,emb2):
        return cos_sim(emb1, emb2)
    
gte_large_emb_model=GteLargeEmbModel()
