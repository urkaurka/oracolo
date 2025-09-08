import json
from pprint import pprint

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from config_manager import config
from llm_interface import LLMInterface

local_template = """
Sei un esperto studioso di alimentazione, salute ed ambiente.
Ti occupi da anni di scrivere news sui tuoi temi di interesse.
In particolare devi leggere il contesto che ti viene proposto e generare
almeno {question} domande che trovano una risposta puntuale nel contesto.
Utilizza un linguaggio tecnico ma comprensibile.
Se devi utilizzare un termine molto tecnico definiscilo prima.

La risposta deve essere un json che contiene un campo questions che è una
lista stringhe contenenti le domande e null'altro.
Mi raccomnado che non ci siano, prima o dopo, introduzioni,psiegazioni o quant'altro altrimenti
non riesco a decodificare il json

Contesto:
{context}

Domande:
"""


class QuestionCreator:
    def __init__(self, text: str):
        self.text = text
        self.llm_interface = LLMInterface(
            model_name=config.config['llm']['model'],
            prompt=local_template
        )
        # Define the response schema for a JSON list of questions
        response_schemas = [
            ResponseSchema(
                name="questions",
                description="A list of questions as strings"
            )
        ]
        self.parser = StructuredOutputParser.from_response_schemas(response_schemas)

    def make_queries(self, nr_query: int) -> list[str]:
        # Format the prompt with the parser's instructions
        # formatted_prompt = self.llm_interface.prompt_template.format(
        #     context=self.text,
        #     question=str(nr_query)
        # ) + "\n\n" + self.parser.get_format_instructions()

        # # Log the formatted prompt for debugging
        # print(f"Formatted Prompt:\n{formatted_prompt}")

        # Get the raw response from the LLM
        raw_response = self.llm_interface.get_response(
            context=self.text,
            question=str(nr_query)
        )
        return json.loads(raw_response)['questions']
