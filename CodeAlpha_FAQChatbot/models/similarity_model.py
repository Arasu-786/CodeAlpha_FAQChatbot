from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticMatcher:

    def __init__(self, questions):

        self.questions = questions

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.question_embeddings = (
            self.model.encode(
                questions,
                convert_to_numpy=True
            )
        )

    def find_best_match_with_confidence(
        self,
        query
    ):

        query_embedding = (
            self.model.encode(
                [query],
                convert_to_numpy=True
            )
        )

        similarities = cosine_similarity(
            query_embedding,
            self.question_embeddings
        )[0]

        best_index = np.argmax(
            similarities
        )

        best_score = similarities[
            best_index
        ]

        return (
            best_index,
            float(best_score)
        )