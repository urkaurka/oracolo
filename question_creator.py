from langchain.schema import Document
from config_manager import config
from llm_interface import LLMInterface
from langchain.output_parsers import CommaSeparatedListOutputParser

local_template = """
Sei un esperto studioso di alimentazione, salute ed ambiente 
Ti occupi da anni di scrivere news sui tuoi temi di interesse..
In particolare devi leggere il contesto che ti viene proposto e generare
almeno {question} domande che trovano una risposta puntuale nel contesto.
Utilizza un linguaggio tecnico ma comprensibile.
Se devi utilizzare un termine molto tecnico definiscilo prima.

Contesto:
{context}

Domande:
"""


class QuestionCreator:
    def __init__(self, doc: Document):
        self.doc = doc
        self.llm_interface = LLMInterface(
            model_name=config.config['llm']['model'],
            prompt=local_template
        )
        # Use a parser to ensure the output is a list of strings
        self.parser = CommaSeparatedListOutputParser()

    def make_queries(self, nr_query: int) -> list[str]:
        """
        Generate a list of questions based on the document content.

        Args:
            nr_query (int): Number of questions to generate.

        Returns:
            list[str]: A list of generated questions.
        """
        # Get the raw response from the LLM
        raw_response = self.llm_interface.get_response(
            context=self.doc.model_dump()['page_content'],
            question=str(nr_query)
        )

        # Parse the response into a list of strings
        try:
            parsed_response = self.parser.parse(raw_response)
            return parsed_response
        except Exception as e:
            # Log the error and return an empty list if parsing fails
            print(f"Error parsing LLM response: {e}")
            return []
