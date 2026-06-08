import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "faiss_index"


def load_index():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"Index loaded from {INDEX_DIR}/\n")
    return vectorstore


def search(vectorstore, query: str, k: int = 3):
    print(f"Q: {query}\n")
    results = vectorstore.similarity_search(query, k=k)
    for i, r in enumerate(results, 1):
        print(f"--- Chunk {i} [{r.metadata['title']}] ---")
        print(r.page_content[:400].strip())
        print()


def main():
    vectorstore = load_index()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        search(vectorstore, query)
    else:
        print("Interactive mode. Type 'exit' to quit.\n")
        while True:
            query = input("Query: ").strip()
            if query.lower() in ("exit", "quit", ""):
                break
            print()
            search(vectorstore, query)


if __name__ == "__main__":
    main()
