import streamlit as st
from chatbot.faq_chatbot import FAQChatbot

chatbot = FAQChatbot("data/faq.csv")

st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("AI FAQ Chatbot")
st.write("Ask anything about Artificial Intelligence")

question = st.text_input(
    "Ask your question"
)

if st.button("Get Answer"):

    if question.strip():

        response, score = chatbot.get_response_with_confidence(
            question
        )

        st.success(response)

        st.caption(
            f"Confidence Score: {score:.2f}"
        )

    else:

        st.warning(
            "Please enter a question."
        )