import os
import re
import cloudscraper
from bs4 import BeautifulSoup
from dotenv import load_dotenv


from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from streamlit.string_util import clean_text
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever

from langchain_community.document_compressors import FlashrankRerank

from retrieveurls import URLRetriever

load_dotenv()
google_key = os.getenv("GOOGLE_API_KEY")
debug_mode = os.getenv("DEBUG", "False")

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=google_key)

#embeddings = GPT4AllEmbeddings(model="./models/ggml-all-MiniLM-L6-v2-f16.bin", n_ctx=512, n_threads=8)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")



# Load HTML
# loader = AsyncChromiumLoader(["https://www.gcu.ac.uk/study/courses/undergraduate-software-development-glasgow"])
# html = loader.load()

docs_urls = URLRetriever().getLinks(url="https://www.gcu.ac.uk/sitemap.xml")
document = []
docs = []
# creating a document class for each scrapped page. This contains the meta data

from clean_docs import clean_content, clean_doc, get_relevant_content

for url in docs_urls:
    document.append(cloudscraper.create_scraper().get(url).content.decode('utf-8'))
    for doc in document:
        soup = clean_doc(doc)
        text = get_relevant_content(soup)
        text = clean_text(text)
        print(text)
        docs.append(Document(
            metadata={"source": url},
            page_content=text,
        ))

template = """You are an expert University Admissions assistant for Glasgow Caledonian University.
        Use the following pieces of retrieved content to answer a query.
        If you the context allows you to, please provide in depth details regarding queries. 
        If the answer cannot be found in the context, please say you don't know and kindly direct them to the admissions office email (admissions@gcu.ac.uk).
        Keep the tone professional and encouraging. 

        Context: {context}
        Question: {question}
        Answer: """
from process_documents import process_docs

base_retriever = process_docs(embeddings, docs).as_retriever(search_kwargs={"k":10})

compressor = FlashrankRerank()

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

def get_ai_response(query: str):
    prompt = ChatPromptTemplate.from_template(template)

    ragChain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )
    response = ragChain.invoke(query)



    return response































