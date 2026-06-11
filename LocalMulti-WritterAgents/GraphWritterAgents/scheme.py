from pydantic import BaseModel, Field
from typing import List, TypedDict, Optional

class Research(BaseModel):
    key_facts: List[str] = Field(description='7–10 key facts about the topic.')
    key_numbers: List[str] = Field(description='Specific figures and statistics.')
    summary: str = Field(description='A brief summary in 3–6 sentences.')

class Critique(BaseModel):
    approved: bool = Field(description='True if the article is good (score >= 1)')
    score: int = Field(description='Score from 1 to 10.', ge=1, le=10)
    feedback: str = Field(description='What needs to be improved in the article.')

class Article(BaseModel):
    title: str = Field(description='Article title.')
    introduction: str = Field(description='Introductory paragraph, 50–80 words.')
    body: str = Field(description='Main body, 200–250 words.')
    conclusion: str = Field(description='Conclusion, 40–60 words.')

class State(TypedDict):
    topic: str
    max_iterations: int

    research: Optional[Research]

    article: Optional[Article]
    iteration: int

    approved: bool
    score: int
    feedback: str