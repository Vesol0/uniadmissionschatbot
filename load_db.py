import os

from langchain_pinecone import PineconeVectorStore
from openai.types import vector_store
from pinecone import Pinecone

class db:
    def __init__(self, embeddings):
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(name="indexone", host="https://indexone-7l7dpjy.svc.aped-4627-b74a.pinecone.io")

        self.vector_store = PineconeVectorStore(embedding=embeddings, index=index)

    def load(self):
        return vector_store




