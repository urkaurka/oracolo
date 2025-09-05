from langchain.schema import Document

from config_manager import config
from llm_interface import LLMInterface


local_template = """
Sei un esperto che si occupa da anni di progetti europei.
In particolare devi leggere il contesto che ti viene proposto e generare
Almeno {question} domande che trovano una risposta puntuale nel contesto.
Utilizza un linguaggio tecnico ma comprensibile.
Se devi utilizzare un termine molto tecnico definiscilo prima

contesto:
{context}

"""


class QuestionCreator:
    def __init__(self, doc: Document):
        self.doc = doc
        self.llm_interface = LLMInterface(
            model_name=config.config['llm']['model'],
            prompt=local_template)

    def make_queries(self, nr_query: int) -> list[str]:
        return self.llm_interface.get_response(
            context=self.doc.model_dump()['page_content'],
            question=str(nr_query))
