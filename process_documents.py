import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
import streamlit as st
load_dotenv()

#This method performs chunking and transforming chunks into vector embeddings before inserting them into vector database
def process_docs(embeddings, docs):
    with (st.spinner('Loading Embeddings...')): # alert user that embeddings are loading

        vector_store = Chroma( # define vector storage (chroma)
           collection_name="embeddings_rag",
            embedding_function=embeddings,
            persist_directory="./chromadb.db", ## store locally
            chroma_cloud_api_key=os.getenv("CHROMA_CLOUD_API_KEY"),

        )
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        batch_size = 5000 # this batch was here previously when we tried adding all of the courses to the vector databae.

        text_splitter = RecursiveCharacterTextSplitter( # set up text splitter
             chunk_size=800,  # chunk size
            chunk_overlap=200,  # chunk overlap
            separators=["\n\n", "\n", ".", " ", ""]
        )
        all_splits = text_splitter.split_documents(docs) # split documents into chunks
        #
        st.toast(f"Split blog post into {len(all_splits)} sub-documents.")

        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i:i + batch_size] # avoid issues with chroma db
            vector_store.add_documents(batch)

        return vector_store # return database object
