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

    def get_documents_by_metadata(self, key: str, value: str) -> list[Document]:
        try:
            results = self.db._collection.get(
                where={key: value},
                include=['documents', 'metadatas']
            )

            documents = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(results['documents'], results['metadatas'])
            ]
            return documents
        except Exception as e:
            print(f"Error retrieving documents with metadata {key}={value}: {str(e)}")
            return []
