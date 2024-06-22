from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from llm import get_llm
from prompt import get_prompt
from entity import ChatReq
import uvicorn


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
    return {"result": "Hello World"}


@app.post("/chatbot/")
async def chatbot(chat_req: ChatReq):
    global history
    question = chat_req.question

    prompt = get_prompt()

    chain = prompt | llm | parser

    history = history[-4:]

    response = chain.invoke({"history": history, "question": [
                            HumanMessage(content=question)]})

    history.extend([HumanMessage(content=question),
                   AIMessage(content=response)])

    return {"result": response}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
