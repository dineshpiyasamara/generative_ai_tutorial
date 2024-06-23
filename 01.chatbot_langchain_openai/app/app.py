from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from llm import get_llm
from prompt import get_prompt
from entity import ChatReq
import uvicorn
import os
from logger import logging
from dotenv import load_dotenv

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

parser = StrOutputParser()
logging.info("initialized output parser")


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

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
