from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
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
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
            
        documents = loader.load()
        return self.text_splitter.split_documents(documents)

    def process_large_pdf(self, file_path: str, batch_size: int = 100) -> None:
        """
        Processa un PDF grande in batch per gestire meglio la memoria
        """
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("Il file deve essere un PDF")
            
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Processa il documento in batch
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            chunks = self.text_splitter.split_documents(batch)
            # Qui puoi aggiungere la logica per salvare i chunks
            # Per esempio, passarli al vector store
            yield chunks