from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, SystemMessage

from langfuse import observe, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from SubAgent import DoCritic, DoResearcher, DoWritter, store, predefined_trace_id

import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

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

llm = ChatOpenAI(
    model='google/gemma-4-e2b',
    temperature=0,
    api_key=os.getenv("LOCAL_API_KEY"),
    base_url="http://localhost:1234/v1",
)

tools = [DoCritic, DoResearcher, DoWritter] + tools_mcp
manager_agent = create_agent(
    model=llm, 
    tools=tools
) 

langfuse = get_client()
langfuse_handler = CallbackHandler()

langfuse_prompt_manager = langfuse.get_prompt(
    "Prompt for manager agent", 
    label="latest"
)
prompt_manager = langfuse_prompt_manager.get_langchain_prompt()

async def ask_agent(question: str) -> str:
    store.clear()

    with langfuse.start_as_current_observation(
        name="manager-agent",
        trace_context={"trace_id": predefined_trace_id}
    ) as span:
        span.update(input=question)

        span.score_trace(
            name="user-feedback",
            value=1,
            data_type="NUMERIC",
            comment="This was correct, thank you"
        )

        response = await manager_agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=prompt_manager),
                    HumanMessage(content=question)
                ]
            },
            config={"callbacks": [langfuse_handler]}
        )
        result = response["messages"][-1].content

        span.update(output=result)

    return result