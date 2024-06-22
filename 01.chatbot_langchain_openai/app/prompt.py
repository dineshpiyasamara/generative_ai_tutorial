from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage


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
