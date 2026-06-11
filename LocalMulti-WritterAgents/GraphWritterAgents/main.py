from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

from route import ask_agent
from scheme import Article

app = FastAPI()

class GenerateRequest(BaseModel):
    query: str

class GenerateResponse(BaseModel):
    response: Article

@app.post("/generate", response_model=GenerateResponse)
def generate(question: GenerateRequest):
    response = ask_agent(question.query)

    return GenerateResponse(response=response)

if __name__ == "__main__":
    uvicorn.run(
        'main:app',
        host='localhost',
        port=8000,
        reload=True
    )