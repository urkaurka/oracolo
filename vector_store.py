from langchain_chroma import Chroma
from langchain.schema import Document
from chromadb.config import Settings

from embeddings_manager import EmbeddingsManager


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings_manager = EmbeddingsManager()
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings_manager.embeddings,
            client_settings=Settings(
                anonymized_telemetry=False
            )
        )

    def add_documents(self, documents: list[Document]):
        """Add documents to the vector store"""
        self.db.add_documents(documents)

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        """Search for documents most similar to the query"""
        return self.db.similarity_search(query, k=k)

    def clear_database(self) -> None:
        """Delete all documents from the database by recreating the collection"""
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings_manager.embeddings
        )
