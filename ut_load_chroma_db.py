import logging
import shutil
from time import perf_counter

from vector_store import VectorStore
from document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [PID: %(process)d - %(filename)s %(funcName)s] - %(message)s",
        level=logging.INFO)
    logger.info("start")

    chroma_dir = "./chroma_db"
    file_pdf = "../resources/Documento VN.pdf"

    shutil.rmtree(chroma_dir)

    ts = perf_counter()
    vs = VectorStore(persist_directory=chroma_dir)
    logger.info(f"VectorStore creation: {perf_counter()-ts} sec")

    ts = perf_counter()
    dp = DocumentProcessor()
    logger.info(f"DocumentProcessor creation: {perf_counter()-ts} sec")

    ts = perf_counter()
    results = list(dp.process_large_pdf(file_pdf))
    for enne, doc in enumerate(results):
        if enne % 10 == 0:
            logger.info(f'step {enne} / {len(results)}')
        vs.add_documents([doc, ])

    logger.info(f"add_documents for {enne+1} docs {perf_counter()-ts}")
