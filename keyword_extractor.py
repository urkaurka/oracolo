import json

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from config_manager import config
from llm_interface import LLMInterface

local_template = """
{context}
Sei un esperto studioso di alimentazione, salute ed ambiente.
Ti occupi da anni di scrivere news sui tuoi temi di interesse.
Devi trasformare delle domande di utenti del tuo sito di blog in un elenco di keyword per ricercare url
su un motore di ricerca a keyword.

In particolare devi leggere la question che ti viene passata e proporre un elenco di keyword
Genera sempre e solo keyword in italiano
*********
Question:
{question}
*********
La risposta deve essere un json che contiene un campo keywords che è una lista delle keyword che hai individuato e null'altro.
Controlla almeno tre volte  che la risposta che dai sia un json ben strutturato che contiene un campo keywords che a sua volta contiene una lista di stringhe che sono le keyword che hai generato.
Mi raccomando utilizza sempre quando possibile parole in italiano.
"""


class KeywordExtractor:
    def __init__(self, question: str):
        self.question = question
        self.llm_interface = LLMInterface(
            model_name=config.config['llm']['model'],
            prompt=local_template
        )
        # Define the response schema for a JSON list of questions
        response_schemas = [
            ResponseSchema(
                name="keywords",
                description="A list of keywords as strings"
            )
        ]
        self.parser = StructuredOutputParser.from_response_schemas(response_schemas)

    def extract_keywords(self) -> list[str]:
        # formatted_prompt = self.llm_interface.prompt_template.format(
        #     context=self.text,
        #     question=str(nr_query)
        # )  # + "\n\n" + self.parser.get_format_instructions()
        # print(f"Formatted Prompt:\n{formatted_prompt}")

        # Get the raw response from the LLM
        raw_response = self.llm_interface.get_response(
            context='',
            question=self.question
        )
        if raw_response.startswith("```"):
            raw_response = raw_response[3:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]

        return json.loads(raw_response)['keywords']
