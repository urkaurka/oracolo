from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingsManager:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts"""
        return self.embeddings.embed_documents(texts)
