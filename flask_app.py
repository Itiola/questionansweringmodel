from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
from transformers import pipeline

app = Flask(__name__)
app.config["CACHE_TYPE"] = "SimpleCache"
cache = Cache(app)

# Load different models for selection
models = {
    "distilbert": pipeline("question-answering", model="distilbert-base-cased-distilled-squad"),
    "bert": pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad"),
    "roberta": pipeline("question-answering", model="deepset/roberta-base-squad2"),
}

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_answer", methods=["POST"])
def get_answer():
    try:
        data = request.get_json()
        question = data.get("question")
        context = data.get("context")
        model_choice = data.get("model", "distilbert")

        if not question or not context:
            return jsonify({"error": "Invalid input"}), 400
        
        cache_key = f"{question}_{context}_{model_choice}"
        if cache.get(cache_key):
            return jsonify(cache.get(cache_key))
        
        # Summarize context if too long
        if len(context.split()) > 300:
            context = summarizer(context, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        
        qa_pipeline = models.get(model_choice, models["distilbert"])
        result = qa_pipeline(question=question, context=context)
        
        response = {
            "answer": result["answer"],
            "confidence": round(result["score"] * 100, 2)
        }
        cache.set(cache_key, response, timeout=600)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
