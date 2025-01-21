from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        question = request.form["question"]
        context = request.form["context"]
        if question and context:
            result = qa_pipeline(question=question, context=context)
            answer = result["answer"]
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
