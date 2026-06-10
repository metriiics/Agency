from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, SystemMessage

from SubAgent import DoCritic, DoResearcher, DoWritter, store

import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from prompts import prompt_manager

async def get_tools_mcp():
    client = MultiServerMCPClient(
        {
            "planner": {
                "command": "python",
                "args": ["mcp-server.py"],
                "transport": "stdio"
            }
        }
    )
    tools = await client.get_tools()
    return tools

tools_mcp = asyncio.run(get_tools_mcp())

llm = ChatGroq(
    model='llama-3.1-8b-instant',
    temperature=0,
    api_key=os.getenv("GROQ_KEY")
)

tools = [DoCritic, DoResearcher, DoWritter] + tools_mcp
manager_agent = create_agent(
    model=llm, 
    tools=tools
) 

async def ask_agent(question: str) -> str:
    store.clear()

    response = await manager_agent.ainvoke(
        {
            "messages": [
                SystemMessage(content=prompt_manager),
                HumanMessage(content=question)
            ]
        }
    )

    return response["messages"][-1].content