from flask import Flask, request, jsonify
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
import faiss
import numpy as np
import os

app = Flask(__name__)

# ✅ Use Hugging Face embeddings instead of OpenAI
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model


# ✅ Load FAISS index and metadata
def load_vector_index():
    storage_path = "./storage"

    if not os.path.exists(storage_path):
        raise ValueError("Storage directory not found. Did you run `ingest.py` first?")

    if not os.path.exists(os.path.join(storage_path, "faiss.index")):
        raise ValueError("FAISS index not found. Did you run `ingest.py` first?")

    # ✅ Load FAISS index
    faiss_index = faiss.read_index(os.path.join(storage_path, "faiss.index"))
    vector_store = FaissVectorStore(faiss_index)

    # ✅ Load storage context
    storage_context = StorageContext.from_defaults(persist_dir=storage_path, vector_store=vector_store)

    # ✅ Load index from storage
    index = load_index_from_storage(storage_context)
    return index, faiss_index  # ✅ Return FAISS index separately


try:
    index, faiss_index = load_vector_index()
except ValueError as e:
    print(f"Error loading index: {e}")
    index = None
    faiss_index = None


@app.route("/query", methods=["POST"])
def query():
    if index is None or faiss_index is None:
        return jsonify({"error": "Index is not loaded. Please run `ingest.py` first."}), 500

    user_query = request.json.get("query", "").lower()
    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # ✅ Retrieve documents (top 10 similar results)
    retriever = index.as_retriever(similarity_top_k=10)
    documents = retriever.retrieve(user_query)

    if not documents:
        return jsonify({"error": "No relevant information found."}), 404

    # ✅ Compute similarity scores manually
    query_embedding = embed_model.get_text_embedding(user_query)  # Convert query to embedding
    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)  # Reshape for FAISS

    distances, indices = faiss_index.search(query_embedding, len(documents))  # ✅ Get similarity scores

    # ✅ Sort results by similarity score (higher score = better match)
    results_with_scores = []
    for idx, doc in enumerate(documents):
        score = round(1 - distances[0][idx], 4)  # ✅ Convert FAISS distance to similarity score
        results_with_scores.append((doc, score))

    results_with_scores.sort(key=lambda x: x[1], reverse=True)  # ✅ Sort descending by score

    # ✅ Ensure each result is unique & limit to top 10
    unique_results = []
    seen_units = set()

    for doc, score in results_with_scores:
        unit_name = doc.metadata.get("unit_name", "unknown").lower()

        if unit_name not in seen_units and len(unique_results) < 10:  # ✅ Limit to top 10 unique results
            seen_units.add(unit_name)
            unique_results.append({
                "Faction": doc.metadata.get("faction", "Unknown"),
                "Unit": doc.metadata.get("unit_name", "Unknown"),
                "Description": doc.text,
                "Index Score": score  # ✅ Include similarity score
            })

    if not unique_results:
        return jsonify({"error": "No relevant units found."}), 404

    # ✅ Save results to `answer.txt`
    with open("answer.txt", "a", encoding="utf-8") as file:
        file.write(f"Query: {user_query}\n")
        for result in unique_results:
            file.write(
                f"{result['Faction']} - {result['Unit']} (Score: {result['Index Score']}): {result['Description']}\n")
        file.write("\n" + "=" * 50 + "\n")  # Separator for better readability

    return jsonify(unique_results)


if __name__ == "__main__":
    app.run(debug=True)