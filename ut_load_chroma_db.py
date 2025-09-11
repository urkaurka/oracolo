import csv
import logging
from time import perf_counter
from pathlib import Path
import shutil

from dibi.utils import get_connection

from settings import DB
from langchain.schema import Document
from vector_store import VectorStore
from document_processor import DocumentProcessor
from config_manager import config

logger = logging.getLogger(__name__)


def load_from_csv(path_csv: Path):
    with path_csv.open() as si:
        reader = csv.reader(si, delimiter='|')
        return [tuple(row) for row in reader]


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

    threshold = 900
    ts = perf_counter()

    conn = get_connection(DB)
    with conn.cursor() as cursor:
        cmd = """
          select external_id,
                 title,
                 text
          from blog_post
          order by external_id"""
        cursor.execute(cmd)
        rows = list(cursor.fetchall())
        for enne, (external_id, title, text) in enumerate(rows):
            if enne % 1 == 0:
                logger.info(f"{enne} / {len(rows)}")

            full_text = f"{title}\n\n{text}"
            partial = ''
            chunks = []
            for sent in dp.split_into_sentences(full_text):
                sent = sent.strip()
                if len(partial) > threshold:
                    chunks.append(partial)
                    partial = ''
                else:
                    partial += f"\n{sent}"
            if partial:
                if len(partial) > 400:
                    chunks.append(partial)
                else:
                    chunks[-1] += f"\n{partial}"

            for emme, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={"from_file": f"row_{external_id:08d}", "pos": emme})
                vs.add_documents([doc, ])
