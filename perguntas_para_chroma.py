import json
import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def load_jsonl(filepath):
    """Load JSONL file and yield each json object."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    # Path to your JSONL file
    jsonl_path = "your_data.jsonl"

    # Prepare data for embedding
    texts = []
    metadatas = []

    print("Loading and preparing data...")
    for entry in load_jsonl(jsonl_path):
        # Combine relevant fields as context for embedding
        combined_text = f"{entry['question']} {entry['objeto']} {entry['answer']} {entry['valor']}"
        texts.append(combined_text)

        # Keep metadata such as ID and question for reference
        metadatas.append({
            "id": entry.get("id", ""),
            "question": entry.get("question", ""),
            "answer": entry.get("answer", ""),
            "objeto": entry.get("objeto", ""),
            "valor": entry.get("valor", "")
        })

    print(f"Loaded {len(texts)} documents.")

    # Create OpenAI embedding model instance
    embedding = OpenAIEmbeddings()

    # Create (or load) Chroma vector store
    persist_directory = "./chroma_db"

    if os.path.exists(persist_directory):
        print("Loading existing Chroma database...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        print("Creating new Chroma database and adding documents...")
        vectorstore = Chroma.from_texts(texts=texts, embedding=embedding, metadatas=metadatas, persist_directory=persist_directory)
        vectorstore.persist()
        print("Chroma database saved.")

if __name__ == "__main__":
    main()
