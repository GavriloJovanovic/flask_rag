import os
import faiss
import numpy as np
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore
from llama_index.core.settings import Settings

# ✅ List of predefined unit types and factions
FACTIONS = ["Western Europe", "Byzantines", "Arabs", "Persians", "Indians"]
UNIT_TYPES = ["ranged", "cavalry", "siege", "infantry", "archer", "knight", "crossbowman", "horse archer"]

# ✅ Load data from text file and enhance metadata
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", required_exts=[".txt"])
    documents = reader.load_data()

    enhanced_docs = []
    for doc in documents:
        text = doc.text.lower()

        # ✅ Extract metadata from text
        faction = next((f for f in FACTIONS if f.lower() in text), "Other")
        unit_type = next((t for t in UNIT_TYPES if t in text), "Other")

        # ✅ Store metadata
        doc.metadata = {
            "faction": faction,
            "unit_type": unit_type
        }
        enhanced_docs.append(doc)

    return enhanced_docs

def create_vector_index():
    documents = load_data()

    # ✅ Use Hugging Face embedding model (local & free)
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model  # ✅ Apply globally

    # ✅ Define embedding dimension
    dummy_text = "test sentence"
    dummy_vector = embed_model.get_text_embedding(dummy_text)
    embedding_dimension = len(dummy_vector)

    # ✅ Initialize FAISS index
    faiss_index = faiss.IndexFlatL2(embedding_dimension)
    vector_store = FaissVectorStore(faiss_index)

    # ✅ Ensure storage directory exists
    storage_path = "./storage"
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    # ✅ Create structured metadata storage
    docstore = SimpleDocumentStore()
    index_store = SimpleIndexStore()

    storage_context = StorageContext.from_defaults(
        persist_dir=storage_path,
        vector_store=vector_store,
        docstore=docstore,
        index_store=index_store
    )

    # ✅ Create LlamaIndex with metadata-enhanced documents
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    # ✅ Persist metadata and FAISS index
    index.storage_context.persist()
    faiss.write_index(faiss_index, os.path.join(storage_path, "faiss.index"))

    print("✅ Hybrid Vector Index created successfully!")

if __name__ == "__main__":
    create_vector_index()
