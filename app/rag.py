import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from openai import OpenAI

from app.prompts import SYSTEM_PROMPT, USER_TEMPLATE

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "faiss_index"

SIMILARITY_THRESHOLD = 1.0  # L2 distance; <1.0 ≈ cosine similarity >0.5
TOP_K = 3

NOT_FOUND = "The knowledge base does not contain information on this topic."

_vectorstore = None
_llm = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
        _vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def get_llm() -> OpenAI:
    global _llm
    if _llm is None:
        folder_id = os.getenv("YANDEX_FOLDER_ID")
        _llm = OpenAI(
            api_key=os.getenv("YANDEX_API_KEY"),
            base_url="https://llm.api.cloud.yandex.net/v1",
            default_headers={"x-folder-id": folder_id},
        )
    return _llm


def ask(question: str) -> dict:
    vs = get_vectorstore()
    results = vs.similarity_search_with_score(question, k=TOP_K)
    relevant = [(doc, score) for doc, score in results if score < SIMILARITY_THRESHOLD]

    if not relevant:
        return {"answer": NOT_FOUND, "sources": []}

    context_parts = []
    sources = []
    for doc, _score in relevant:
        title = doc.metadata.get("title", "Unknown")
        context_parts.append(f"[{title}]\n{doc.page_content.strip()}")
        if title not in sources:
            sources.append(title)

    context = "\n\n".join(context_parts)
    user_message = USER_TEMPLATE.format(context=context, question=question)

    folder_id = os.getenv("YANDEX_FOLDER_ID")
    llm = get_llm()
    response = llm.chat.completions.create(
        model=f"gpt://{folder_id}/deepseek-v4-flash/latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    if NOT_FOUND in answer:
        return {"answer": answer, "sources": []}
    return {"answer": answer, "sources": sources}
