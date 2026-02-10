import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

llm = LLM(model='gpt-3.5-turbo')

qa_agent = Agent(
    role="Question Answering Chatbot",
    goal="Provide accurate, concise answers to user question: {question}",
    backstory="You serve as a reliable guide to help users navigate information overload and find trustworthy answers.",
    tools=[],
    llm=llm
)

qa_task = Task(
    description="Answer the user’s question, ensuring clarity, accuracy, and relevance.",
    expected_output="A clear and concise answer addressing the user's question with supporting details if necessary.",
    agent=qa_agent
)

crew = Crew(
    agents=[qa_agent],
    tasks=[qa_task],
    verbose=True
)

# result = crew.kickoff(inputs={'question': 'what is machine learning'})
# print(result)

result = crew.kickoff(inputs={'question': 'what is codeprolk'})
print(result)
