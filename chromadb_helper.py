import chromadb
from embedding_model import gte_large_emb_model
import pandas as pd


class ChromaVectorDB:
    def __init__(self, emb_model):
        self.__client__ = chromadb.PersistentClient(path="./chroma_db")
        self.__emb_model__ = emb_model
        self.eco_invest_data_path = "data/eco_invest_data_for_embedding.csv"
        self.eco_invest_collection = self.collection = (
            self.__client__.get_or_create_collection(
                name="sample", metadata={"hnsw:space": "cosine"}
            )
        )

    def update_eco_invest_collection(self):
        df = pd.read_csv(self.eco_invest_data_path)
        metadatas = df.to_dict(orient="records")
        texts = df["reference_product"].to_list()
        for i, text in enumerate(texts):

            embedding = self.__emb_model__.get_embeddings([text])

            self.eco_invest_collection.upsert(
                ids=[str(i)], embeddings=[embedding[0]], metadatas=[metadatas[i]]
            )

        print("updated collection")

    def get_matching_items(self, query_text, max_n_items):
        query_embedding = self.__emb_model__.get_embeddings([query_text])
        query_result = self.collection.query(
            query_embeddings=[query_embedding[0]],
            n_results=max_n_items,
            # where={"metadata_field": "is_equal_to_this"},
            # where_document={"$contains":"search_string"}
        )

        meta_records = query_result.get("metadatas", [])
        if len(meta_records) > 0:

            return meta_records[0]  # list of records
        return meta_records


chroma_db_instant = ChromaVectorDB(emb_model=gte_large_emb_model)
