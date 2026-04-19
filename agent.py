
import os
import time
import cloudscraper
from dotenv import load_dotenv
from datasets import Dataset
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from flashrank import Ranker


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from clean_documents import document_cleaner
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
load_dotenv()
google_key = os.getenv("GOOGLE_API_KEY")
debug_mode = os.getenv("DEBUG", "False")

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", api_key=google_key) # define model (gemini-3.1-flash-lite-preview)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en") # set embedding model

"""
Original Document Pre-procesing and ingestion method
"""
#ßdocs_urls = URLRetriever().getLinks(url="https://www.gcu.ac.uk/sitemap.xml") # initial site map loader
docs_urls = ["https://www.gcu.ac.uk/study/courses/undergraduate-computing-glasgow", "https://www.gcu.ac.uk/study/courses/undergraduate-software-development-glasgow", "https://www.gcu.ac.uk/study/courses/undergraduate-ai-and-data-science-glasgow"]
document = []
docs = []

scraper = cloudscraper.create_scraper() # create scraper object

for url in docs_urls:
    print(f"Scraping {url}")
    html_content = scraper.get(url).content.decode('utf-8') # get html content
    cleaned_data = document_cleaner(html_content) # clean html content

    clean_text = f""" 
    Course Title: {cleaned_data.get("Title", '')}
    Overview: {cleaned_data.get("Overview", '')}
    Modules Taught: {cleaned_data.get("Modules", '')}
    Course Details & Fees: {cleaned_data.get("Details", '')}

"""

    docs.append(Document( # append to documents with a langchain document object.
        metadata={"source": url, "course_title": cleaned_data.get("Title", ''), "overview": cleaned_data.get("Overview", ''), "modules": cleaned_data.get("Modules", ''), "details": cleaned_data.get("Details", '')},
        page_content=clean_text
    ))

# define prompt template/system prompt
template = """You are an expert University Admissions assistant for Glasgow Caledonian University.
        Use the following pieces of retrieved content to answer a query.
        If you don't know the answer please say you don't know and kindly direct them to the admissions office email (admissions@gcu.ac.uk).
        Keep the tone professional and encouraging. 

        Context: {context}
        Question: {question}
        Answer: """
from process_documents import process_docs

semantic_retriever = process_docs(embeddings, docs).as_retriever(search_kwargs={"k":10}) ## semantic retriever
lexical_retriever = BM25Retriever.from_documents(docs) # BM25 sparse retriever
lexical_retriever.k = 6

hybrid_retriever = EnsembleRetriever( # hybrid retriever configuration
    retrievers=[semantic_retriever, lexical_retriever],
    weights=[0.6, 0.4]
)
from flashrank import Ranker

Ranker(model_name="ms-marco-MultiBERT-L-12")
compressor = FlashrankRerank() #reranker

retriever = ContextualCompressionRetriever( # reranks the retrieval results from hybrid retriever
    base_compressor=compressor,
    base_retriever=hybrid_retriever
)


def get_ai_response(query: str):
    prompt = ChatPromptTemplate.from_template(template) #set system prompt
    ragChain = ( # define rag chain
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )
    response = ragChain.invoke(query) # get llm response
    return response # return response to pass through to front end.

# evaluation
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def test_ragas_evaluation():
    print("Starting  RAGAS evaluation...")
    # three test cases due to limited api calls
    test_cases = [
        {
            "question": "What are the tuition fees for Software development",
            "truth": "The tuition fees are: Home: £1,820, Rest of uk: £9,535, International: £15,200, *Scottish student tuition fees are subject to confirmation by the Scottish government and may change once confirmed. "
        },
        {
            "question": "What grades do I need for year entry into Computing?",
            "truth": "The Standard Entry requirement is Scottish Higher: BBCC ore equivalent incl Highetr in Math, Appl of Maths, or Computing, A level: CCC or equivalent incl A level in Maths or Computing, Ucas Tariff: 96. Minimum Entry Requirements: UCAS Tarrif: 90, Scottish Higher: BCCC or equivalent incl Higher in Maths, Appl of Maths, or Computing, A Level: CCC or equivalent incl A level in Maths, Appl of Maths or Computing"
        },
        {
            "question":"Is there a placement year for AI and Data Science",
            "truth": "Yes, students on this course have the option of a year-long industrial placement or the possible to study abroad through turing exchanges. They will also be encouraged to take part in activities with the international association for the exchange of students for Technical Experience (IAESTE). "
        }
    ]

    questions =[]
    answers =[]
    context_lists = []
    truths = []
    print("Fetching response from Gemini")

    for i, test in enumerate(test_cases): # for each text case
        agent_answer = get_ai_response(test["question"])
        retrieved_docs = retriever.invoke(test['question'])
        context = [doc.page_content for doc in retrieved_docs]

        questions.append(test['question'])
        answers.append(agent_answer)
        context_lists.append(context)
        truths.append(test['truth'])
        print("Pausing for 2 seconds to avoid rate limits")
        time.sleep(3)

    data_dict = { # create dictionary
        "question": questions,
        "answer": answers,
        "contexts": context_lists,
        "ground_truth": truths
    }
    eval_dataset = Dataset.from_dict(data_dict) # convert dict to dataset
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") # necessary for huggingface.

    print("Setting up HuggingFace evaluation model")
    hf_llm = HuggingFaceEndpoint( # configuration for the evaluator model
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        max_new_tokens=2048,
        do_sample=False,
        huggingfacehub_api_token=hf_token,

    )
    eval_chat_model = ChatHuggingFace(llm=hf_llm)
    ragas_llm = LangchainLLMWrapper(eval_chat_model)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    print("Running RAGAS metrics")
    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=True
    )
    df = result.to_pandas() # create dataframe for pandas
    df.to_csv('data_semantic_retrieval.csv', header=True) # save to csv file for visualizations

    print("\n--- Evaluation Results ---")
    print(result)

if __name__ == "__main__":
    test_ragas_evaluation()






















