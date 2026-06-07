import pandas as pd
from models.similarity_model import SemanticMatcher

class FAQChatbot:

    def __init__(self, faq_path):

        self.data = pd.read_csv(faq_path)

        self.data = self.data.dropna(
            subset=["question", "answer"]
        )

        self.matcher = SemanticMatcher(
            self.data["question"].tolist()
        )

        self.fallback_message = (
            "Sorry, I couldn't find a relevant answer."
        )

    def get_response(
        self,
        user_input,
        threshold=0.3
    ):

        index, score = (
            self.matcher
            .find_best_match_with_confidence(
                user_input
            )
        )

        if score < threshold:

            return self.fallback_message

        return self.data.iloc[index]["answer"]

    def get_response_with_confidence(
        self,
        user_input,
        threshold=0.3
    ):

        index, score = (
            self.matcher
            .find_best_match_with_confidence(
                user_input
            )
        )

        if score < threshold:

            return (
                self.fallback_message,
                0.0
            )

        return (
            self.data.iloc[index]["answer"],
            score
        )