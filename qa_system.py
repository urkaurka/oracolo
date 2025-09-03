from .vector_store import VectorStore
from .llm_interface import LLMInterface

class QASystem:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = LLMInterface(model_name="llama2")  # or other model supported by Ollama
    
    def answer_question(self, question: str) -> str:
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(question, k=3)
        
        # Prepare context
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Get response from LLM
        response = self.llm.get_response(context=context, question=question)
        return response