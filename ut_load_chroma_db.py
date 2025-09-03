import logging

from vectore_store import VectorStore
from document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [PID: %(process)d - %(filename)s %(funcName)s] - %(message)s",
        level=logging.INFO)
    logger.info("start")

    vs = VectorStore(persist_directory="./chroma_db")
    vs.clear_database()

    dp = DocumentProcessor()
