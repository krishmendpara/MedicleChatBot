# app.py

from flask import Flask, render_template, request, jsonify
from connect_memory_with_llm import qa_chain

app = Flask(__name__)


# =====================
# MAIN PAGE
# =====================

@app.route("/")
def home():
    return render_template("index.html")


# =====================
# CHAT API
# =====================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_message = data.get("message")

    if not user_message:
        return jsonify({"answer": "Please ask a medical question."})

    try:

        response = qa_chain.invoke({
            "input": user_message
        })

        answer = response["answer"]

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "answer": "Error processing request."
        })


# =====================
# RUN SERVER
# =====================

if __name__ == "__main__":
    app.run(debug=True)