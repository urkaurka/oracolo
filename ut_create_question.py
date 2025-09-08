import logging
import random
from pprint import pprint


from vector_store import VectorStore
from question_creator import QuestionCreator

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [PID: %(process)d - %(filename)s %(funcName)s] - %(message)s",
        level=logging.INFO)
    logger.info("start")

    chroma_dir = "./chroma_db"
    vs = VectorStore(persist_directory=chroma_dir)
    doc_ids = vs.get_all_document_ids()
    random.shuffle(doc_ids)

    doc = vs.get_document_by_id(doc_ids[0])
    qc = QuestionCreator(doc)
    results = qc.make_queries(2)
    pprint(results)
