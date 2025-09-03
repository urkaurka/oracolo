from langchain.schema import Document

class QuestionCreator:
    def __init__(self, doc: Document):
        self.doc = doc
        self.llm_interface = LLMInterface

    def make_queries(self, nr_query: str) -> list[str]:
