from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from typing import List, Generator
from langchain.schema import Document


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_and_split(self, file_path: str) -> List[Document]:
        """Load a document and split it into chunks"""
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        documents = loader.load()
        return self.text_splitter.split_documents(documents)

    def process_large_pdf(self, file_path: str, batch_size: int = 100) -> Generator[List[Document], None, None]:
        """
        Process a large PDF in batches to better manage memory

        Args:
            file_path: path to the PDF file
            batch_size: size of each document batch

        Yields:
            List[Document]: batch of processed documents
        """
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("File must be a PDF")

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            chunks = self.text_splitter.split_documents(batch)
            yield chunks
