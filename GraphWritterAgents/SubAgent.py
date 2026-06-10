from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from prompts import prompt_critic, prompt_researcher, prompt_writter, prompt_struct
from scheme import Article, Critique, Research
 
store = {}

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

tools = asyncio.run(get_tools_mcp())

llm = ChatOpenAI(
    model='llama-3.1-8b-instant',
    temperature=0,
    api_key=os.getenv("LOCAL_API_KEY"),
    base_url="http://localhost:1234/v1",
)

writter_agent = create_agent(
    model=llm, 
    response_format=Article
)
research_agent = create_agent(
    model=llm, 
    tools=tools,
    response_format=Research
)
critic_agent = create_agent(
    model=llm,
    response_format=Critique 
)

@tool
async def DoResearcher(question: str) -> str:
    """ Researches the topic: searches for facts on the internet and structures them. Invoke it first. """
    response = await research_agent.ainvoke(
        {
            "messages": [
                SystemMessage(content=prompt_researcher),
                HumanMessage(content=question)
            ]
        }
    )

    output = response["structured_response"]

    store['research'] = output
    result = f'Собрано {len(output.key_facts)} фактов. Резюме: {output.summary}'
    return result

@tool
async def DoWritter(content: str, feedback: str = '') -> str:
    """ Writes an article based on the research. If feedback is provided, incorporates it. Invoke it after do_research. """

    res = store.get('research')
    if not res:
        return 'Ошибка: сначала вызови do_research'  

    facts = '\n'.join(f'- {f}' for f in res.key_facts)
    nums = '\n'.join(f'- {n}' for n in res.key_numbers)
    fb_part = f'\n\nУчти замечания: {feedback}' if feedback else ''  

    response = await writter_agent.ainvoke(
        {
            "messages": [
                SystemMessage(content=prompt_writter),
                HumanMessage(content=(
                    f'Тема: {content}\n\nРезюме: {res.summary}\n\n'
                    f'Факты:\n{facts}\n\nЦифры:\n{nums}{fb_part}'
            ))
            ]
        }
    )

    output = response["structured_response"]

    store['article'] = output

    result = f'Статья написана. Заголовок: {output.title}'
    return result

@tool
async def DoCritic(content: str) -> str:
    """ Evaluates the article and returns feedback. Invoke it after write_article. """

    res = store.get('article')
    if not res:
        return 'Ошибка: сначала вызови write_article'

    text = f'{res.title}\n\n{res.introduction}\n\n{res.body}\n\n{res.conclusion}'

    response = await critic_agent.ainvoke(
        {
            "messages": [
                SystemMessage(content=prompt_critic),
                HumanMessage(content=f'Оцени статью на тему "{content}":\n\n{text}')
            ]
        }
    )

    output = response["structured_response"]

    store['critique'] = output

    result = f'Оценка: {output.score}/10. Одобрена: {output.approved}. Замечания: {output.feedback}'
    return result