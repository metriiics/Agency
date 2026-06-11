from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

import os
from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP("Application")

tavily_client = TavilyClient(
    api_key=os.getenv('TAVILY_KEY')
)

@mcp.tool()
def tool_search_browser(question: str):
    """ browser search """
    response = tavily_client.search(question)
    return response

@mcp.tool()
def tool_create_md(file_name: str, content: str) -> None:
    """ create markdown file """

    with open(f'archive/{file_name}.md', 'w', encoding="utf-8") as file:
        file.write(content)
    return f"File created 'archive/{file_name}.md'"

if __name__ == "__main__":
    mcp.run(transport="stdio")