from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate


def get_prompt():
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are an intelligent chatbot. Answer the following question."),
            MessagesPlaceholder(variable_name="history"),
            MessagesPlaceholder(variable_name="question")
        ]
    )
    return prompt


def get_rag_prompt():
    message = """
    Answer this question using the provided context only.

    {question}

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages([("human", message)])
    return prompt
