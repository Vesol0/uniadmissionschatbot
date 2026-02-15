from langchain_core.vectorstores import InMemoryVectorStore
import streamlit as st

def process_docs(embeddings, docs) -> InMemoryVectorStore:
    with st.spinner('Loading Embeddings...'):
        vector_store = InMemoryVectorStore(embeddings)
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
