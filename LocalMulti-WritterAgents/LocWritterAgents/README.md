

```
@observe()
async def ask_agent(question: str) -> str:
    store.clear()

    with propagate_attributes(
        trace_name="user-query-processing",
        session_id="session-123",
        user_id="user-587"
    ):
        langfuse_handler = CallbackHandler()

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

        langfuse.set_current_trace_io(
            input={"query": question},
            output={"response": result}
        )

    return result
```