import os

from langchain.agents.middleware import dynamic_prompt, ModelRequest, HumanInTheLoopMiddleware, SummarizationMiddleware
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

os.environ["OPENAI_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAAN2K5kcZrG8A7o1-4o8GfpE1xF8Z2Z4g"



model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")


embeddings = GPT4AllEmbeddings(model="./models/ggml-all-MiniLM-L6-v2-f16.bin", n_ctx=512, n_threads=8)
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from process_documents import process_docs

# Load HTML
loader = AsyncChromiumLoader(["https://www.gcu.ac.uk/study/courses/undergraduate-software-development-glasgow"])
html = loader.load()
template = """You are an expert University Admissions assistant.
        Use the following pieces of retrieved content to answer a query.
        If you the context allows you to, please provide in depth details regarding queries. 
        If the answer cannot be found in the context, please say you don't know and kindly direct them to the admissions office email.
        Keep the tone professional and encouraging. 

        Context: {context}
        Question: {question}
        Answer: """
from process_documents import process_docs

retriever = process_docs(embeddings, html).as_retriever(search_kwargs={"k":3})

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

os.environ["PINECONE_API_KEY"] = "pcsk_77iHq3_CZQc9Evk9oUGKFZacc4N9shmJs1ME684RboeUumshDNPo1UxGu1eKLJm1oxFZ2C"


#links = URLRetriever().getLinks("https://www.gcu.ac.uk/sitemap.xml")

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer



from langchain.agents import create_agent































