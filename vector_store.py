from langchain.vectorstores import Chroma
from .embeddings_manager import EmbeddingsManager
from typing import List
from langchain.schema import Document


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings_manager = EmbeddingsManager()
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings_manager.embeddings
        )

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        self.db.add_documents(documents)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Search for documents most similar to the query"""
        return self.db.similarity_search(query, k=k)

    def clear_database(self) -> None:
        """Delete all documents from the database"""
        self.db._collection.delete(where={})
        self.db.persist()
