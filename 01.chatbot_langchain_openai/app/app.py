from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from llm import get_llm
from prompt import get_prompt
from entity import ChatReq
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title="Chatbot",
    version="1.0",
    decsription="A simple chatbot API server using OpenAI and LangChain"
)

history = []
llm = get_llm()
parser = StrOutputParser()


@app.get("/")
async def index():
    return {"result": "Welcome to CodePRO LK"}


@app.post("api/chatbot/")
async def chatbot(chat_req: ChatReq):
    global history

    response_data = {
        "status": "fail",
        "response": ""
    }

    try:
        question = chat_req.question

        prompt = get_prompt()

        chain = prompt | llm | parser

        history = history[-4:]

        response = chain.invoke({"history": history, "question": [
                                HumanMessage(content=question)]})

        history.extend([HumanMessage(content=question),
                        AIMessage(content=response)])

        response_data["response"] = response
        response_data["status"] = "success"
        return response_data
    except Exception as e:
        response_data["response"] = str(e)
        return response_data

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
