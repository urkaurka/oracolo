import logging
from collections import Counter
from pprint import pprint
import random


from vector_store import VectorStore
from question_creator import QuestionCreator

logger = logging.getLogger(__name__)


def dump_all_from_file(vs: VectorStore) -> list[tuple[str, int]]:
    value_cnt = Counter()
    for id in vs.get_all_document_ids():
        doc = vs.get_document_by_id(id)
        for key, value in doc.metadata.items():
            if key == "from_file":
                value_cnt[value] += 1
    return value_cnt.most_common()


def get_full_text_from_file(vs: VectorStore, from_file: str) -> str:
    documents = vs.get_documents_by_metadata(key="from_file", value=from_file)
    sorted_docuements = sorted(documents,
                               key=lambda x: x.metadata['pos'],
                               reverse=False)
    return '\n'.join([doc.page_content for doc in sorted_docuements]).strip()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [PID: %(process)d - %(filename)s %(funcName)s] - %(message)s",
        level=logging.INFO)
    logger.info("start")

    chroma_dir = "./chroma_db"
    vs = VectorStore(persist_directory=chroma_dir)

    results = dump_all_from_file(vs)
    while 1:
        random.shuffle(results)
        from_file = results[0][0]
        logger.info(f"load from row nr {from_file}")

        full_text = get_full_text_from_file(vs, from_file)
        logger.info(f"title: {full_text.split('\n')[0].strip()}")

        qc = QuestionCreator(full_text)
        queries = qc.make_queries(5)
        for query in queries:
            print(f"{query=}")

            import pdb; pdb.set_trace()

            docs = vs.similarity_search(query, k=3)
            pprint([(doc.metadata['from_file'], doc.metadata['pos']) for doc in docs])
            print()

        input('premi invio')
