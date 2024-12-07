from rank_bm25 import BM25Okapi
import numpy as np

class Retriever:
    def __init__(self, docs: list[str]):
        self.docs = docs

        tokenized_docs = [doc.lower().split(" ") for doc in docs]

        self.bm25 = BM25Okapi(tokenized_docs)

    def get_docs(self, query, n=3) -> list[str]:
        bm25_scores = self._get_bm25_scores(query)

        sorted_indices = np.argsort(bm25_scores)

        actual_n = -min(n, len(self.docs) - 1)

        return [self.docs[i] for i in sorted_indices[:actual_n]]
    
    def _get_bm25_scores(self, query):
        tokenized_query = query.lower().split(' ')

        return self.bm25.get_scores(tokenized_query)

