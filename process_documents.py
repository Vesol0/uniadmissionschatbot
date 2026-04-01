import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
import streamlit as st
load_dotenv()

def process_docs(embeddings, docs):
    with (st.spinner('Loading Embeddings...')):

        vector_store = Chroma(
           collection_name="uni_documents",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain.db",
            chroma_cloud_api_key=os.getenv("CHROMA_CLOUD_API_KEY"),

        )
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        batch_size = 5000

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # chunk size (characters)
            chunk_overlap=100,  # chunk overlap (characters)
            separators=["\n\n", "\n", ".", " "]
        )
        all_splits = text_splitter.split_documents(docs)

        st.toast(f"Split blog post into {len(all_splits)} sub-documents.")

        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i:i + batch_size]
            vector_store.add_documents(batch)
        #vector_store = FAISS.from_documents(all_splits,embeddings)
        #vector_store.add_documents(all_splits)
        return vector_store
