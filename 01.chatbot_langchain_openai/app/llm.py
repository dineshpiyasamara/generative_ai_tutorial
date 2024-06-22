import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI


def get_llm():
    load_dotenv()

    os.environ['AZURE_OPENAI_API_KEY'] = os.getenv("AZURE_OPENAI_API_KEY")

    llm = AzureChatOpenAI(
        model="gpt-3.5-turbo",
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_DEPLOYMENT")
    )
    return llm
