from pprint import pprint
import logging
import json
from time import perf_counter
from pathlib import Path
import shutil


from vector_store import VectorStore
from document_processor import DocumentProcessor
from config_manager import config

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [PID: %(process)d - %(filename)s %(funcName)s] - %(message)s",
        level=logging.INFO)
    logger.info("start")

    ts = perf_counter()
    chroma_path = Path(config.chroma_dir)
    if chroma_path.exists():
        shutil.rmtree(config.chroma_dir)
    chroma_path.mkdir()
    logger.info("reset chroma db")

    vs = VectorStore(persist_directory=config.chroma_dir)
    logger.info(f"VectorStore creation: {perf_counter()-ts} sec")

    ts = perf_counter()
    dp = DocumentProcessor()
    logger.info(f"DocumentProcessor creation: {perf_counter()-ts} sec")

    json_dir = Path("../web_scraping/scraper_project/scraper_project/_data")
    ts = perf_counter()
    for enne, file_json in enumerate(json_dir.iterdir()):
        print(f"{enne: 2d} - {file_json.name}")
        documents = dp.load_and_split_json(file_json)
        for enne, doc in enumerate(documents):
            doc.metadata = {
                "from_file": file_json.name,
                "pos": enne
            }
        vs.add_documents(documents)
