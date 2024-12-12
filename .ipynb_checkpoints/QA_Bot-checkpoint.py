import streamlit as st
from transformers import pipeline

def main():
    st.title("Question Answering Bot")

    # Load the pre-trained model
    question_answerer = pipeline("question-answering")

    # Get the user's question
    user_question = st.text_input("Ask your question:")

    # Define the context (replace with your desired context)
    context = """
    Economics is the social science that studies the production, distribution, and consumption of goods and services. It examines how individuals, businesses, governments, and nations make choices to allocate limited resources.
    """

    if user_question:
        # Perform question answering
        answer = question_answerer(question=user_question, context=context)

        # Display the answer
        st.write("Answer:")
        st.write(answer['answer'])

if __name__ == "__main__":
    main()