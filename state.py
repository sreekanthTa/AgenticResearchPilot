from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator


class ResearchInput(BaseModel):
    question: str = Field(min_length=3, max_length=50)

    @field_validator("question", mode="before")
    def clean_question(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Question must be txt")

        value = value.strip()

        if len(value) < 3:
            raise ValueError("Question must contain at least 3 characters")

        return value


class ResearchPilotState(TypedDict):
    question: str
    is_research: bool
    messages: Annotated[list, add_messages]
    tasks: list[str]
    search_results: list[dict]
    analysis: str
    citation_checks: list[dict]
    revision_count: int
    report: str
