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

from retrieveurls import URLRetriever

load_dotenv()
google_key = os.getenv("GOOGLE_API_KEY")
debug_mode = os.getenv("DEBUG", "False")




model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=google_key)


embeddings = GPT4AllEmbeddings(model="./models/ggml-all-MiniLM-L6-v2-f16.bin", n_ctx=512, n_threads=8)
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from process_documents import process_docs

# Load HTML
# loader = AsyncChromiumLoader(["https://www.gcu.ac.uk/study/courses/undergraduate-software-development-glasgow"])
# html = loader.load()

docs_urls = URLRetriever().getLinks(url="https://www.gcu.ac.uk/sitemap.xml")
document = []
for url in docs_urls:
    document.append(cloudscraper.create_scraper().get(url).content)
docs = ''
for doc in document:
    docs += ' '+doc.decode('utf-8')




#document = cloudscraper.create_scraper().get("https://www.gcu.ac.uk/study/courses/undergraduate-software-development-glasgow")


soup = BeautifulSoup(docs, 'html.parser')

print(soup.get_text().strip())








doc = Document(
    page_content=soup.get_text().strip(),
    metadata={"source": "https://www.gcu.ac.uk/study/courses/undergraduate-software-development-glasgow"},
)


template = """You are an expert University Admissions assistant for Glasgow Caledonian University.
        Use the following pieces of retrieved content to answer a query.
        If you the context allows you to, please provide in depth details regarding queries. 
        If the answer cannot be found in the context, please say you don't know and kindly direct them to the admissions office email (admissions@gcu.ac.uk).
        Keep the tone professional and encouraging. 

        Context: {context}
        Question: {question}
        Answer: """
from process_documents import process_docs

retriever = process_docs(embeddings, [doc]).as_retriever(search_kwargs={"k":10})

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





import getpass


from langchain_google_genai import GoogleGenerativeAIEmbeddings



#links = URLRetriever().getLinks("https://www.gcu.ac.uk/sitemap.xml")

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer



from langchain.agents import create_agent































