import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()  # load HF_API_TOKEN, PINECONE_API_KEY, MONGODB_URI etc. from .env

from rag.rag_chain import chat
from rag.vectorstore import count
from ingestion.indexer import ingest_all, ingest_single

app = Flask(__name__)
CORS(app)  # allow requests from the React frontend

# on startup: if Pinecone is empty, pull all PDFs from MongoDB and index them
with app.app_context():
    indexed = count()
    if indexed == 0:
        print("Vectorstore is empty — running ingestion on startup...")
        try:
            ingest_all()
            print(f"Startup ingestion done. {count()} chunks indexed.")
        except Exception as e:
            print(f"Startup ingestion failed: {e}")


@app.route("/health", methods=["GET"])
def health():
    # used by Express proxy to check if Flask is alive
    return jsonify({"status": "ok", "docs_indexed": count()})


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "question is required"}), 400

    try:
        result = chat(question, n_results=data.get("n_results", 5))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ingest", methods=["POST"])
def ingest_endpoint():
    # called automatically by Express when faculty uploads a new file
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    title = data.get("title", "Untitled")
    subject = data.get("subject", "General")

    if not url:
        return jsonify({"error": "url is required"}), 400

    try:
        ingest_single(url, title, subject)
        return jsonify({"message": f"'{title}' ingested successfully", "docs_indexed": count()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ingest-all", methods=["POST"])
def ingest_all_endpoint():
    # manually re-index everything — useful if Pinecone data is lost
    try:
        ingest_all()
        return jsonify({"message": "All materials ingested", "docs_indexed": count()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
