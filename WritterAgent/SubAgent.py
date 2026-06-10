from langchain_core.tools import tool
from langchain_groq import ChatGroq
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

llm = ChatGroq(
    model='llama-3.1-8b-instant',
    temperature=0,
    api_key=os.getenv("GROQ_KEY")
)

writter_agent = create_agent(
    model=llm, 
)
research_agent = create_agent(
    model=llm, 
    tools=tools
)
critic_agent = create_agent(
    model=llm, 
)

structed_writter = llm.with_structured_output(Article)
structed_researcher = llm.with_structured_output(Research)
structed_critic = llm.with_structured_output(Critique)

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

    output = response["messages"][-1].content

    StructedResponse = await structed_researcher.ainvoke(
        f"""
            {prompt_struct}

            Ответ:
            {output}
        """
    )

    store['research'] = StructedResponse
    result = f'Собрано {len(StructedResponse.key_facts)} фактов. Резюме: {StructedResponse.summary}'
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

    output = response["messages"][-1].content

    StructedResponse = await structed_writter.ainvoke(
        f"""
            {prompt_struct}

            Ответ:
            {output}
        """
    )
    store['article'] = StructedResponse

    result = f'Статья написана. Заголовок: {StructedResponse.title}'
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

    output = response["messages"][-1].content

    StructedResponse = await structed_critic.ainvoke(
        f"""
            {prompt_struct}

            Ответ:
            {output}
        """
    )
    store['critique'] = StructedResponse

    result = f'Оценка: {StructedResponse.score}/10. Одобрена: {StructedResponse.approved}. Замечания: {StructedResponse.feedback}'
    return result