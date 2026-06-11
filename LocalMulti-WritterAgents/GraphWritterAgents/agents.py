from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from langfuse import Langfuse, observe, propagate_attributes
from langfuse.langchain import CallbackHandler

import os
from dotenv import load_dotenv
load_dotenv()

from scheme import Article, Critique, Research, State

search = TavilySearchResults(max_results=5)

llm_writter = ChatOpenAI(
    model='google/gemma-4-e2b',
    temperature=0.7,
    api_key=os.getenv("LOCAL_API_KEY"),
    base_url="http://localhost:1234/v1",
)

llm = ChatOpenAI(
    model='google/gemma-4-e2b',
    temperature=0,
    api_key=os.getenv("LOCAL_API_KEY"),
    base_url="http://localhost:1234/v1",
)

writter_agent = create_agent(
    model=llm_writter,
    response_format=Article
)
research_agent = create_agent(
    model=llm,
    response_format=Research
)
critic_agent = create_agent(
    model=llm,
    response_format=Critique
)

langfuse = Langfuse()
langfuse_handler = CallbackHandler()

langfuse_prompt_researcher = langfuse.get_prompt(
    "Prompt for researcher agent",
    label="latest"
)
prompt_researcher = langfuse_prompt_researcher.get_langchain_prompt()

langfuse_prompt_writter = langfuse.get_prompt(
    "Prompt for writter agent",
    label="latest"
)
prompt_writter = langfuse_prompt_writter.get_langchain_prompt()

langfuse_prompt_critic = langfuse.get_prompt(
    "Prompt for critic agent",
    label="latest"
)
prompt_critic = langfuse_prompt_critic.get_langchain_prompt()

@observe()
def Researcher(state: State) -> dict:
    """ Researches the topic: searches for facts on the internet and structures them. Invoke it first. """

    raw = search.invoke(state["topic"])
    context = '\n\n'.join(f'{r["url"]}\n{r["content"]}' for r in raw)

    with propagate_attributes(
        trace_name="user-query-processing-researcher",
        session_id='session_888',
        user_id='user-144'
    ):
        response = research_agent.invoke(
            {
                "messages": [
                    SystemMessage(content=prompt_researcher),
                    HumanMessage(content=f'Тема: {state["topic"]}\n\nИсточники:\n{context}')
                ]
            },
            config={"callbacks": [langfuse_handler]}
        )

        output = response["structured_response"]

        langfuse.set_current_trace_io(
            input={"query": f'Тема: {state["topic"]}\n\nИсточники:\n{context}'},
            output={"response": output}
        )

    return {"research": output}


def Writter(state: State) -> dict:
    """ Writes an article based on the research. If feedback is provided, incorporates it. Invoke it after do_research. """
    iteration = state.get('iteration', 0)
    res = state['research']

    facts = '\n'.join(f'- {f}' for f in res.key_facts)
    nums = '\n'.join(f'- {n}' for n in res.key_numbers)

    revision = ''
    if iteration > 0:
        prev = state['article']
        revision = (
            f'\n\nLast version(title): {prev.title}'
            f'\nEditor remarks: {state["feedback"]}'
            f'\nPlease note the comments in the new version'
        )

    with propagate_attributes(
        trace_name="user-query-processing-writter",
        session_id='session_888',
        user_id='user-144'
    ):
        response = writter_agent.invoke(
            {
                "messages": [
                    SystemMessage(content=prompt_writter),
                    HumanMessage(content=(
                        f'Тема: {state["topic"]}\n\n'
                        f'Резюме: {res.summary}\n\n'
                        f'Факты:\n{facts}\n\nЦифры:\n{nums}'
                        f'{revision}'
                ))
                ]
            },
            config={"callbacks": [langfuse_handler]}
        )

        output = response["structured_response"]

        langfuse.set_current_trace_io(
            input={"query":  {
                "theme": state["topic"],
                "summary": res.summary,
                "facts": facts,
                "nums": nums,
                "revision": revision
            }},
            output={"response": output}
        )

    return {'article': output, 'iteration': iteration + 1}


def Critic(state: State) -> dict:
    """ Evaluates the article and returns feedback. Invoke it after write_article. """

    res = state['article']

    text = f'{res.title}\n\n{res.introduction}\n\n{res.body}\n\n{res.conclusion}'

    with propagate_attributes(
        trace_name="user-query-processing-critic",
        session_id='session_888',
        user_id='user-144'
    ):
        response = critic_agent.invoke(
            {
                "messages": [
                    SystemMessage(content=prompt_critic),
                    HumanMessage(content=f'Оцени статью на тему {state["topic"]}:\n\n{text}')
                ]
            },
            config={"callbacks": [langfuse_handler]}
        )

        output = response["structured_response"]

        langfuse.set_current_trace_io(
            input={"query":            {
                "content": state['article'],
                "text": text,
            }},
            output={"response": output}
        )

    return {"approved": output.approved, "score": output.score, "feedback": output.feedback}