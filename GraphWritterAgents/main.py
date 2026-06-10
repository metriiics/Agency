from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

from agent import ask_agent

app = FastAPI()

class GenerateRequest(BaseModel):
    quest: str

class GenerateResponse(BaseModel):
    resp: str

@app.post("/generate", response_model=GenerateResponse)
async def generate(question: GenerateRequest):
    response = await ask_agent(question.quest)

    return GenerateResponse(resp=response)

if __name__ == "__main__":
    uvicorn.run(
        'main:app',
        host='localhost',
        port=8000,
        reload=True
    )