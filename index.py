import os
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI

# os.environ['OPENAI_API_KEY'] = "<your_key>"

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
agent_executer = create_csv_agent(llm, 'data.csv', verbose=True, allow_dangerous_code=True)

agent_executer.invoke("How many total wins are there?")


