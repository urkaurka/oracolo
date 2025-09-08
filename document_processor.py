import json
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document

from typing import List


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_and_split(self, file_path: str) -> list[Document]:
        """Load a document and split it into chunks"""
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        documents = loader.load()
        return self.text_splitter.split_documents(documents)

    def process_large_pdf(self, file_path: str, batch_size: int = 100):
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("File must be a PDF")

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            for chunk in self.text_splitter.split_documents(batch):
                yield chunk

    def load_and_split_json(self, file_path: Path) -> List[Document]:
        if not file_path.name.lower().endswith('.json'):
            raise ValueError("File must be a JSON file")
        data = json.load(file_path.open())

        if "text" not in data:
            raise ValueError(f"The JSON file {file_path} does not contain a 'text' field")
        text = f'{data["title"]}\n\n{data["text"]}'
        documents = [Document(page_content=text)]
        return self.text_splitter.split_documents(documents)
