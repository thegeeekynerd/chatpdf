import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------- EXTRACT TEXT ----------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            txt = page.extract_text()
            if txt:
                text += txt
    return text

# ---------------------- CHUNK TEXT ----------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

# ---------------------- VECTOR STORE ----------------------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# ---------------------- GEMINI ANSWER ----------------------
def answer_question_with_gemini(context, question):
    prompt = f"""
You are an AI assistant. Answer the question using **only** the given context.
If the answer is not found, reply: "I cannot find the answer in the provided PDF."

Context:
{context}

Question:
{question}

Answer:
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ---------------------- HANDLE USER QUERY ----------------------
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    context = "\n\n".join([d.page_content for d in docs])

    answer = answer_question_with_gemini(context, user_question)

    st.write("### ✅ Answer:")
    st.write(answer)

    with st.expander("📌 Show Retrieved Context"):
        for i, doc in enumerate(docs):
            st.write(f"**Chunk {i+1}:**")
            st.write(doc.page_content)
            st.write("---")


# ---------------------- UI ----------------------
def main():
    st.set_page_config("ChatPDF - AI RAG Demo")
    st.header("💬 Chat with Your PDF (Gemini RAG)")

    user_question = st.text_input("Ask something from your PDFs")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 PDF Upload")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing PDFs..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success("✅ Vector Store Created!")
                else:
                    st.error("Please upload at least one PDF.")

if __name__ == "__main__":
    main()
