import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

llm = LLM(model='gpt-3.5-turbo')


@tool("add_tool")
def add(num1: float, num2: float) -> float:
    """Adds two numbers and returns the result."""
    return num1 + num2


@tool("subtract_tool")
def subtract(num1: float, num2: float) -> float:
    """Subtracts the second number from the first and returns the result."""
    return num1 - num2


@tool("multiply_tool")
def multiply(num1: float, num2: float) -> float:
    """Multiplies two numbers and returns the result."""
    return num1 * num2


math_agent = Agent(
    role="Math Expert",
    goal="Solve math problems by performing arithmetic operations",
    backstory="A skilled mathematician who can perform addition, subtraction, multiplication.",
    tools=[add, subtract, multiply],
    llm=llm
)

task = Task(
    description="Calculate the result of {question}",
    expected_output="A single number as the final result.",
    agent=math_agent
)

crew = Crew(
    agents=[math_agent],
    tasks=[task],
    verbose=True
)

# result = crew.kickoff(
#     inputs={'question': 'Add 5 and 7'})
# print(result)

# result = crew.kickoff(
#     inputs={'question': 'Add 5 and 7, then multiply it by 5.'})
# print(result)
