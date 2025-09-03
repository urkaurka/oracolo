from langchain.llms import Ollama
from langchain import PromptTemplate, LLMChain

class LLMInterface:
    def __init__(self, model_name: str = "llama2"):
        self.llm = Ollama(model=model_name)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Basandoti sul seguente contesto, rispondi alla domanda. 
            Usa solo le informazioni fornite nel contesto.
            
            Contesto:
            {context}
            
            Domanda: {question}
            
            Risposta:"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def get_response(self, context: str, question: str) -> str:
        """Ottiene una risposta dall'LLM dato un contesto e una domanda"""
        return self.chain.run(context=context, question=question)