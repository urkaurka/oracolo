from langchain.llms import Ollama
from langchain import PromptTemplate, LLMChain

class LLMInterface:
    def __init__(self, model_name: str = "llama2"):
        self.llm = Ollama(model=model_name)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Based on the following context, answer the question.
            Use only information provided in the context.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def get_response(self, context: str, question: str) -> str:
        """Get a response from the LLM given a context and question"""
        return self.chain.run(context=context, question=question)