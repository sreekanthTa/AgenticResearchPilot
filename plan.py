from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from model import llm_modl
from state import ResearchPilotState


class ResearchLLMop(BaseModel):
    tasks: list[str] = Field(
        min_length=3,
        max_length=5,
        description="Three to five specific web research tasks.",
    )


sys_prompt = """
You are the research planner for ResearchPilot.
Break the user's research question into 3 to 5
specific, independent web research tasks.
Do not answer the research question.
"""


llm_with_struct = llm_modl.with_structured_output(
    ResearchLLMop,
    method="json_schema",
)


def plan(state: ResearchPilotState) -> dict:
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=state["question"]),
    ]

    response = llm_with_struct.invoke(messages)

    return {"tasks": response.tasks, "revision_count": 0}
