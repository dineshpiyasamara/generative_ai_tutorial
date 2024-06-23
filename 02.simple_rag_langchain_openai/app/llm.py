import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_openai import OpenAIEmbeddings


def get_llm():
    load_dotenv()
    os.environ['AZURE_OPENAI_API_KEY'] = os.getenv("AZURE_OPENAI_API_KEY")
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

    llm = AzureChatOpenAI(
        model="gpt-3.5-turbo",
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
        max_tokens=50
    )

    # # Set OpenAI API key
    # os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

    # # Initialize the ChatOpenAI model
    # llm = ChatOpenAI(
    #     model="gpt-3.5-turbo",
    #     temperature=0,
    #     max_tokens=50
    # )

    return llm


def get_embedding():

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    return embedding_model
