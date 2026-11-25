import os
import streamlit as st
from dotenv import load_dotenv

from rag_engine import (
    load_and_preprocess_pdf,
    split_documents,
    init_vector_store,
    generate_rag_answer,
)

load_dotenv()

st.set_page_config(
    page_title="RAG Corporate Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG-асистент нормативної документації")
st.caption("Лабораторна робота №7 | Варіант 12 | ChromaDB + Qwen + Groq")

PDF_PATH = "academic_regulations.pdf"
env_key = os.getenv("GROK_API", "")

with st.sidebar:
    st.header("⚙️ Налаштування")

    groq_key = st.text_input(
        "Groq API Key",
        value=env_key,
        type="password",
    )

    if env_key:
        st.success("API-ключ завантажено з .env")
    else:
        st.warning("API-ключ не знайдено у .env")

    st.divider()

    uploaded_pdf = st.file_uploader(
        "Завантажити PDF-документ",
        type=["pdf"],
    )

    reindex = st.button("🔄 Переіндексувати документ")


@st.cache_resource(show_spinner=False)
def load_database():
    raw_docs = load_and_preprocess_pdf(PDF_PATH)
    chunks = split_documents(raw_docs, 800, 150)
    db = init_vector_store(chunks, force_reload=False)
    return db, len(chunks)


if uploaded_pdf is not None:
    with open(PDF_PATH, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    st.cache_resource.clear()
    st.info("PDF завантажено. Натисніть «Переіндексувати документ».")

if reindex:
    with st.spinner("Переіндексація документа..."):
        raw_docs = load_and_preprocess_pdf(PDF_PATH)
        chunks = split_documents(raw_docs, 800, 150)
        init_vector_store(chunks, force_reload=True)
        st.cache_resource.clear()

    st.success("Векторну базу оновлено!")

if os.path.exists(PDF_PATH):
    with st.spinner("Завантаження векторної бази..."):
        vector_db, chunk_count = load_database()

    st.success(f"База знань готова! Фрагментів: {chunk_count}")
else:
    st.error("Файл academic_regulations.pdf не знайдено!")
    st.stop()

st.subheader("💬 Поставте запитання до документа")

question = st.text_area(
    "Ваше запитання:",
    value=(
        "Які умови та підстави передбачені для відрахування "
        "та поновлення здобувачів вищої освіти?"
    ),
    height=110,
)

if st.button("🔍 Отримати відповідь", type="primary"):
    if not groq_key:
        st.error("Вкажіть Groq API Key!")
    elif not question.strip():
        st.warning("Введіть запитання!")
    else:
        with st.spinner("Пошук інформації та генерація відповіді..."):
            try:
                answer, retrieved = generate_rag_answer(
                    vector_db,
                    question,
                    groq_key,
                    top_k=5,
                )

                col1, col2 = st.columns([6, 4])

                with col1:
                    st.subheader("🤖 Відповідь Qwen")
                    st.markdown(answer)

                with col2:
                    st.subheader("📄 Знайдені першоджерела")

                    for i, (doc, distance) in enumerate(retrieved, 1):
                        page = doc.metadata.get("page", 0) + 1
                        similarity = 1.0 - distance

                        with st.expander(
                            f"Фрагмент {i} | Стор. {page} | "
                            f"Similarity: {similarity:.3f}"
                        ):
                            st.write(doc.page_content)

            except Exception as e:
                st.error(f"Помилка: {e}")
