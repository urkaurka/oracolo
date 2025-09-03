import logging
import random

from vector_store import VectorStore

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
