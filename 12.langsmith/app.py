from fastapi import FastAPI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.environ.get("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"] = os.environ.get("LANGSMITH_PROJECT")
os.environ["LANGSMITH_ENDPOINT"] = os.environ.get("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")

app = FastAPI()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


loader = PyPDFLoader("data/codeprolk.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=50)
splits = text_splitter.split_documents(docs)


vectorstore = Chroma.from_documents(
    splits,
    embedding=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)


message = """
Answer this question using the provided context only.

Question:
{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


class Request(BaseModel):
    query: str


@app.post('/query/')
def predict(req: Request):
    response = chain.invoke(req.query)
    return {'response': response}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
