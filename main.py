from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate

import os

os.environ["OPENAI_API_KEY"] = ""
# Identify PDFs to Load

pdf1 = "George Ugwu CV.pdf"


# PDF Loader 

def load_pdf(files):
    pages = []
    for file in files: 
        loader = PyPDFLoader(file)
        pages += loader.load_and_split()
    return pages


# Load in the PDFs as `pdf_pages` 
pdf_pages = load_pdf([pdf1])

len(pdf_pages)


# Print out the first 100 characters
print(pdf_pages[0].page_content[:100])

# print(pdf_pages[205].page_content[:100])

# Print the metadata
print(pdf_pages[0].metadata)

# print(pdf_pages[205].metadata)

# Define the embeddings model
embeddings = OpenAIEmbeddings()
db = Weaviate.from_documents(pdf_pages, embeddings, weaviate_url="http://localhost:8080", by_text=False)

# query = "What is my names?"
# docs = db.similarity_search(query)

# print(docs[0].metadata)
# print(docs[0].page_content[:100])
# # print(docs[1].metadata)
# # print(docs[1].page_content[:100])

# query = "What are my skills ?"
# docs = db.similarity_search(query)

# print(docs[0].metadata)
# print(docs[0].page_content[:100])
# # print(docs[1].metadata)
# # print(docs[1].page_content[:100])

# set the weaviate database as the retiever
retriever = db.as_retriever()
# define our prompt as the RAG prompt from LangChain's prompt hub
prompt = hub.pull("rlm/rag-prompt")
# Set the LLM to use ChatOpenAI 3.5-Turbo
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain.invoke("what is the name on the cv")
