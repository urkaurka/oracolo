import logging
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from config_manager import config

logger = logging.getLogger(__name__)

rag_template = """
    Based on the following context, answer the question.
    Use only information provided in the context.

    Context:
    {context}

    Question: {question}

    Answer:"""


class LLMInterface:
    def __init__(self, model_name: str = config.config['llm']['model'], prompt=rag_template):
        self.llm = Ollama(model=model_name)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt
        )
        # Create a runnable sequence instead of LLMChain
        self.chain = self.prompt_template | self.llm

    def get_response(self, context: str, question: str) -> str:
        formatted_prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        logger.info(f"Prompt sent to LLM:\n{formatted_prompt}")

        response = self.chain.invoke({"context": context, "question": question})
        return response
