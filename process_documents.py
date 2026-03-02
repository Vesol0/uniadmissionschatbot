import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
import streamlit as st
load_dotenv()

def process_docs(embeddings, docs):
    with st.spinner('Loading Embeddings...'):
        vector_store = InMemoryVectorStore(embeddings)
        # vector_store = Chroma(
        #     collection_name="uni_documents",
        #     embedding_function=embeddings,
        #     persist_directory="./chroma_langchain.db",
        #     chroma_cloud_api_key=os.getenv("CHROMA_CLOUD_API_KEY"),
        #
        # )
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # chunk size (characters)
            chunk_overlap=100,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        all_splits = text_splitter.split_documents(docs)


        st.toast(f"Split blog post into {len(all_splits)} sub-documents.")
        vector_store.add_documents(all_splits)
        return vector_store
