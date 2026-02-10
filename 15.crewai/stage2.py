import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import PDFSearchTool

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

llm = LLM(model='gpt-3.5-turbo')

pdf_rag_tool = PDFSearchTool(pdf='codeprolk.pdf')

Researcher = Agent(
    role="Information Research Assistant",
    goal="Extract relevant information from documents to support user query: {query}.",
    backstory="You are skilled at analyzing documents and finding key insights quickly and accurately.",
    tools=[pdf_rag_tool]
)

Writer = Agent(
    role="Content Generation Assistant",
    goal="Create clear, engaging, and well-structured written related to query: {query}.",
    backstory="You are passionate about turning ideas and facts into compelling, easy-to-read content for any audience.",
    tools=[]
)

research_task = Task(
    description="Analyze the provided documents and extract key facts, statistics, and insights related to the query.",
    expected_output="A structured list or summary of important findings, categorized by subtopics or themes.",
    agent=Researcher
)

writer_task = Task(
    description="Use the extracted research findings to create a well-written report that explains the query.",
    expected_output="A polished, three-section report in markdown format with an introduction, detailed explanation, and conclusion.",
    output_file='report.md',
    agent=Writer
)

crew = Crew(
    agents=[Researcher, Writer],
    tasks=[research_task, writer_task],
    verbose=True
)

result = crew.kickoff(
    inputs={'query': 'what is codeprolk'})
print(result)

# result = crew.kickoff(
#     inputs={'query': 'what are the courses offered by codeprolk'})
# print(result)
