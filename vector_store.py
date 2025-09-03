import shutil
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
        self.db.add_documents(documents)

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        return self.db.similarity_search(query, k=k)

    def clear_database(self) -> None:
        shutil.rmtree(self.persist_directory)
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings_manager.embeddings
        )

    def get_all_document_ids(self) -> list[str]:
        return self.db._collection.get()["ids"]

    def get_document_by_id(self, doc_id: str) -> Document | None:
        try:
            result = self.db._collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas']
            )

            if result and result['ids']:
                return Document(
                    page_content=result['documents'][0],
                    metadata=result['metadatas'][0]
                )
            return None
        except Exception as e:
            print(f"Error retrieving document {doc_id}: {str(e)}")
            return None
