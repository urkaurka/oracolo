import logging
from time import perf_counter
from pathlib import Path

from vector_store import VectorStore
from document_processor import DocumentProcessor
from config_manager import ConfigManager

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [PID: %(process)d - %(filename)s %(funcName)s] - %(message)s",
        level=logging.INFO)
    logger.info("start")

    # Load configuration
    config = ConfigManager()

    ts = perf_counter()
    vs = VectorStore(persist_directory=config.chroma_dir)
    vs.clear_database()
    logger.info(f"VectorStore creation: {perf_counter()-ts} sec")

    ts = perf_counter()
    dp = DocumentProcessor()
    logger.info(f"DocumentProcessor creation: {perf_counter()-ts} sec")

    # Construct PDF path
    pdf_path = Path(config.pdf_resources_dir) / "Documento VN.pdf"

    ts = perf_counter()
    results = list(dp.process_large_pdf(str(pdf_path)))
    for enne, doc in enumerate(results):
        if enne % 10 == 0:
            logger.info(f'step {enne} / {len(results)}')
        vs.add_documents([doc, ])

    logger.info(f"add_documents for {enne+1} docs {perf_counter()-ts}")
