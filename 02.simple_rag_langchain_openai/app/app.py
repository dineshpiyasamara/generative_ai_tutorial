from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from llm import get_llm, get_embedding
from prompt import get_prompt, get_rag_prompt
from entity import ChatReq
import uvicorn
import os
from logger import logging
from dotenv import load_dotenv
from data import documents

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title="Chatbot",
    version="1.0",
    decsription="A simple chatbot API server using OpenAI and LangChain"
)
logging.info("initialized FastAPI server")

history = []

llm = get_llm()
logging.info("initialized LLM")

embedding_model = get_embedding()
logging.info("initialized Embedding model")

parser = StrOutputParser()
logging.info("initialized output parser")

vectorstore = Chroma.from_documents(
    documents,
    embedding=embedding_model,
)
logging.info("initialized vector store")


@app.get("/")
async def index():
    response_data = {
        "status": "success",
        "response": "Welcome to CodePRO LK"
    }
    return response_data


@app.post("/api/chatbot")
async def chatbot(chat_req: ChatReq):
    global history

    question = chat_req.question
    logging.info(f"Question: {question}")

    response_data = {
        "status": "success",
        "response": ""
    }

    try:
        prompt = get_prompt()
        logging.info("Load prompt template")

        chain = prompt | llm | parser
        logging.info("Create chain")

        history = history[-4:]

        response = chain.invoke({"history": history, "question": [
                                HumanMessage(content=question)]})
        logging.info(f"Response: {response}")

        history.extend([HumanMessage(content=question),
                        AIMessage(content=response)])

        response_data["response"] = response
        return response_data
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        response_data["response"] = str(e)
        response_data["status"] = "fail"
        return response_data


@app.post("/api/simple_rag")
async def simple_rag(chat_req: ChatReq):
    question = chat_req.question
    logging.info(f"Question: {question}")

    response_data = {
        "status": "success",
        "response": ""
    }

    try:
        prompt = get_rag_prompt()
        logging.info("Load prompt template")

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1},
        )
        logging.info("Initialized retriever")

        chain = {"context": retriever,
                 "question": RunnablePassthrough()} | prompt | llm
        logging.info("Create chain")

        response = chain.invoke(question).content
        logging.info(f"Response: {response}")

        response_data["response"] = response
        return response_data
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        response_data["response"] = str(e)
        response_data["status"] = "fail"
        return response_data

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
