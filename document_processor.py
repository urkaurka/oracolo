from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from typing import List
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_and_split(self, file_path: str) -> List[Document]:
        """Carica un documento e lo divide in chunks"""
        loader = TextLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)