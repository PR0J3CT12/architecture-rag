import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

KB_DIR = Path(__file__).parent.parent / "knowledge_base"
INDEX_DIR = Path(__file__).parent.parent / "faiss_index"

CHUNK_SIZE = 2000       # ~500 tokens
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 80   # stay safely under 100 req/min free-tier limit
EMBED_BATCH_DELAY = 65  # seconds between batches


def load_documents():
    loader = DirectoryLoader(
        str(KB_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    docs = loader.load()
    for doc in docs:
        stem = Path(doc.metadata["source"]).stem
        doc.metadata["title"] = stem.replace("_", " ")
    print(f"Loaded {len(docs)} documents")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_index(chunks, embeddings_model):
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    print(f"Generating embeddings: {len(texts)} chunks in {total_batches} batches...")

    start = time.time()
    all_embeddings = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch_num = i // EMBED_BATCH_SIZE + 1
        batch = texts[i : i + EMBED_BATCH_SIZE]
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end=" ", flush=True)
        batch_embeddings = embeddings_model.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
        print("OK")

        if i + EMBED_BATCH_SIZE < len(texts):
            print(f"  Waiting {EMBED_BATCH_DELAY}s (rate limit)...", flush=True)
            time.sleep(EMBED_BATCH_DELAY)

    elapsed = time.time() - start
    print(f"Embeddings done in {elapsed:.1f}s")

    vectorstore = FAISS.from_embeddings(
        list(zip(texts, all_embeddings)),
        embeddings_model,
        metadatas=metadatas,
    )
    return vectorstore, len(chunks), round(elapsed)


def save_index(vectorstore):
    INDEX_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"Index saved to {INDEX_DIR}/")


def run_example_queries(vectorstore):
    queries = [
        "Who is Kael Mortus and what happened to him?",
        "What is the Void Core and what does it do?",
        "What is Synth Flux?",
    ]
    print("\n--- Example queries ---")
    for query in queries:
        print(f"\nQ: {query}")
        results = vectorstore.similarity_search(query, k=2)
        for r in results:
            print(f"  [{r.metadata['title']}] {r.page_content[:200].strip()}...")


def main():
    docs = load_documents()
    chunks = split_documents(docs)

    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    vectorstore, n_chunks, elapsed = build_index(chunks, embeddings_model)
    save_index(vectorstore)
    run_example_queries(vectorstore)

    print(f"\n=== Summary ===")
    print(f"  Model:   models/gemini-embedding-001")
    print(f"  Docs:    {len(docs)}")
    print(f"  Chunks:  {n_chunks}")
    print(f"  Time:    {elapsed}s")
    print(f"  Index:   {INDEX_DIR}/")


if __name__ == "__main__":
    main()
