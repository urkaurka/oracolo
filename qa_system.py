from .vector_store import VectorStore
from .llm_interface import LLMInterface

class QASystem:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = LLMInterface(model_name="llama2")  # o altro modello supportato da Ollama
    
    def answer_question(self, question: str) -> str:
        # Recupera documenti rilevanti
        relevant_docs = self.vector_store.similarity_search(question, k=3)
        
        # Prepara il contesto
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Ottieni risposta dall'LLM
        response = self.llm.get_response(context=context, question=question)
        return response