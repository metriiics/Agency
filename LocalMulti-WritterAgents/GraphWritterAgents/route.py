from langgraph.graph import StateGraph, END

from langfuse import observe, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from agents import Critic, Researcher, Writter
from scheme import State

import os
from dotenv import load_dotenv
load_dotenv()

def router(state: State) -> str:
    if state['approved']:
        return 'end'

    if state['iteration'] >= state['max_iterations']:
        return 'end'

    return 'Writter'

graph = StateGraph(State)

graph.add_node('Researcher', Researcher)
graph.add_node('Writter', Writter)
graph.add_node('Critic', Critic)

graph.set_entry_point('Researcher')
graph.add_edge('Researcher', 'Writter')
graph.add_edge('Writter', 'Critic')
graph.add_conditional_edges('Critic', router, {'Writter': 'Writter', 'end': END})

app_deterministic = graph.compile()

def ask_agent(question: str):

    result = app_deterministic.invoke(
        {
            'topic': question,
            'max_iterations': 3,
            'research': None,
            'article': None,
            'iteration': 0,
            'approved': False,
            'score': 0,
            'feedback': ''
        }
    )

    return result['article']