import streamlit as st
import google.generativeai  as genai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_vector_db(pdf_doc):

    pdf_reader = PdfReader(pdf_doc)
    pages = [page.extract_text() for page in pdf_reader.pages]
    texts = "\n\n".join(pages)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    chunks = text_splitter.split_text(texts)

    embedding_func = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GOOGLE_API_KEY"))
    vectorstore  = Chroma.from_texts(chunks, embedding_func).as_retriever(search_kwargs={"k":5}) 
   
    return vectorstore



# streamlit application
st.set_page_config("RAG system")
st.markdown("<h2 style='text-align: center; color: Black;'>RAG-based Question Answering System</h2>", unsafe_allow_html=True)

pdf_doc = st.file_uploader("Upload your PDF", type="pdf")
if pdf_doc is not None:
    user_question = st.text_input("Ask a question from your PDF:")
    if user_question:
        vectorstore = get_vector_db(pdf_doc)

        model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=os.getenv("GOOGLE_API_KEY"),
                             temperature=0.2,convert_system_message_to_human=True)
        qa_chain = RetrievalQA.from_chain_type(
            model,
            retriever=vectorstore,
            return_source_documents=True

        )
                    
        result = qa_chain({"query": user_question})
        st.write(result["result"])















# pdf_doc = st.file_uploader("Upload your PDF", type="pdf")
# if pdf_doc is not None:
#     user_question = st.text_input("Ask a question about your PDF:")
#     if user_question:
#         try:
#             pdf_reader = PdfReader(pdf_doc)

#             pages = [page.extract_text() for page in pdf_reader.pages]
#             texts = "\n\n".join(pages)

#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
#             # context = "\n\n".join(str(page.page_content) for page in pdf_reader.pages)
#             chunks = text_splitter.split_text(texts)
#         except:
#             st.error("Invalid PDF file. Please upload a valid PDF.")


#         # pdf_loader = PyPDFLoader("data/yolo.pdf")
#         # pages = pdf_loader.load_and_split()

#         # text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
#         # context = "\n\n".join(str(p.page_content) for p in pages)
#         # texts = text_splitter.split_text(context)

#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GOOGLE_API_KEY"))
#         vectorstore  = Chroma.from_texts(chunks, embeddings).as_retriever(search_kwargs={"k":5})

#         model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=os.getenv("GOOGLE_API_KEY"),
#                              temperature=0.2,convert_system_message_to_human=True)
#         qa_chain = RetrievalQA.from_chain_type(
#             model,
#             retriever=vectorstore,
#             return_source_documents=True

#         )
                
#         result = qa_chain({"query": user_question})
#         st.write(result["result"])

    