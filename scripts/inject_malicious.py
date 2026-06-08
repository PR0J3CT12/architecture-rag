import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "faiss_index"
MALICIOUS_FILE = Path(__file__).parent.parent / "knowledge_base" / "malicious.txt"


def main():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    print("Loading existing index...")
    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print("Loading malicious document...")
    loader = TextLoader(str(MALICIOUS_FILE), encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["title"] = "malicious"

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"  {len(chunks)} chunk(s) to add")

    vectorstore.add_documents(chunks)
    vectorstore.save_local(str(INDEX_DIR))
    print("Index updated and saved.")


if __name__ == "__main__":
    main()
