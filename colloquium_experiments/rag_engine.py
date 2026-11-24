import os
import re
import shutil
import warnings

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

PDF_PATH = "academic_regulations.pdf"
CHROMA_PERSIST_DIR = "./chroma_db_storage"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

IN_DOMAIN_QUERY = (
    "Які умови та підстави передбачені для відрахування "
    "та поновлення здобувачів вищої освіти?"
)

REFUSAL = "У наданому нормативному документі інформація з цього питання відсутня."


def load_and_preprocess_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Не знайдено PDF: {pdf_path}")

    docs = PyPDFLoader(pdf_path).load()

    for doc in docs:
        text = doc.page_content.replace("\u00a0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        doc.page_content = text.strip()

    docs = [doc for doc in docs if doc.page_content]
    print(f"[+] Завантажено сторінок: {len(docs)}")
    return docs


def split_documents(raw_docs, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(raw_docs)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def init_vector_store(chunks, force_reload=False):
    embeddings = get_embeddings()

    if force_reload and os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)

    if os.path.exists(CHROMA_PERSIST_DIR):
        print("[+] Завантаження існуючої ChromaDB...")
        return Chroma(
            collection_name="university_regulations",
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )

    print(f"[*] Індексація {len(chunks)} фрагментів...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="university_regulations",
        persist_directory=CHROMA_PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("[+] Векторну базу ChromaDB створено.")
    return db


def retrieve_context(vector_db, query, top_k=5):
    results = vector_db.similarity_search_with_score(
        query, k=top_k * 3
    )

    unique_docs = []
    seen = set()

    for doc, distance in results:
        normalized = " ".join(doc.page_content.split())

        if normalized in seen:
            continue

        seen.add(normalized)
        unique_docs.append((doc, distance))

        if len(unique_docs) >= top_k:
            break

    return unique_docs


def generate_rag_answer(vector_db, user_query, groq_api_key, top_k=5):
    retrieved = retrieve_context(vector_db, user_query, top_k)

    context_parts = []

    for doc, distance in retrieved:
        page = doc.metadata.get("page", 0) + 1
        similarity = 1.0 - distance
        context_parts.append(
            f"[Сторінка {page}; схожість {similarity:.3f}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = f"""
Ти — асистент з нормативної документації університету.

Відповідай ТІЛЬКИ на основі наданих фрагментів документа.

Правила:
1. Не вигадуй фактів, нормативних положень або строків.
2. Якщо інформації недостатньо, відповідай точно:
   "{REFUSAL}"
3. Для кожного підтвердженого положення зазначай сторінку
   у форматі (Стор. 12).
4. Відповідай українською мовою.
5. Не використовуй зовнішні знання.
"""

    client = Groq(api_key=groq_api_key)

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"КОНТЕКСТ:\n{context}\n\n"
                    f"ЗАПИТАННЯ:\n{user_query}"
                ),
            },
        ],
    )

    return response.choices[0].message.content, retrieved


def compare_chunking(raw_docs):
    configs = [
        ("A", 300, 0),
        ("B", 800, 150),
        ("C", 800, 400),
        ("D", 2000, 200),
    ]

    embeddings = get_embeddings()
    original_length = sum(len(d.page_content) for d in raw_docs)
    rows = []

    for name, size, overlap in configs:
        print(f"[*] Перевірка конфігурації {name}...")
        chunks = split_documents(raw_docs, size, overlap)

        total_length = sum(len(c.page_content) for c in chunks)
        rr = (
            (total_length - original_length) / original_length * 100
            if original_length else 0
        )

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"experiment_{name.lower()}",
            collection_metadata={"hnsw:space": "cosine"},
        )

        result = db.similarity_search_with_score(
            IN_DOMAIN_QUERY, k=1
        )

        similarity = 1.0 - result[0][1] if result else 0

        rows.append({
            "Конфігурація": name,
            "Розмір": size,
            "Перекриття": overlap,
            "Кількість чанків": len(chunks),
            "RR (%)": round(rr, 2),
            "Similarity": round(similarity, 4),
        })

        del db

    df = pd.DataFrame(rows)
    print("\n=== ПОРІВНЯННЯ ФРАГМЕНТАЦІЇ ===")
    print(df.to_string(index=False))
    df.to_csv("chunking_comparison.csv", index=False, encoding="utf-8-sig")
    return df



def compare_distance_metrics(raw_docs):
    print("\n=== ЕКСПЕРИМЕНТ COSINE VS L2 ===")

    chunks = split_documents(raw_docs, 800, 150)
    embeddings = get_embeddings()
    rows = []

    for metric in ["cosine", "l2"]:
        print(f"\n[*] Створення ChromaDB: {metric}")

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"metric_test_{metric}",
            collection_metadata={"hnsw:space": metric},
        )

        results = db.similarity_search_with_score(
            IN_DOMAIN_QUERY,
            k=5
        )

        print(f"\nМетрика: {metric}")

        for rank, (doc, distance) in enumerate(results, 1):
            page = doc.metadata.get("page", 0) + 1

            rows.append({
                "Metric": metric,
                "Rank": rank,
                "Page": page,
                "Distance": round(float(distance), 6),
            })

            print(
                f"Top-{rank}: "
                f"сторінка {page}, "
                f"distance={distance:.6f}"
            )

        del db

    df = pd.DataFrame(rows)

    print("\n=== ПІДСУМКОВА ТАБЛИЦЯ ===")
    print(df.to_string(index=False))

    df.to_csv(
        "distance_metrics_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[+] Результати збережено.")
    return df


if __name__ == "__main__":
    raw_docs = load_and_preprocess_pdf(PDF_PATH)

    # Експеримент №4: cosine проти L2
    compare_distance_metrics(raw_docs)

    print("\n[+] Експеримент завершено.")