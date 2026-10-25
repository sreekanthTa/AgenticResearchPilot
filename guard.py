from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from model import llm_modl, invoke_structured
from state import ResearchPilotState


class Intent(BaseModel):
    is_research: bool


classifier_llm = llm_modl.with_structured_output(Intent, method="json_schema")


sys_prm = """
You decide whether a user's message is a research question that needs
web searching, or ordinary chit-chat.

Set is_research to true only when the message asks for factual
information that would require researching sources (for example
"what is X", "compare X and Y", "the latest news about Z").

Set is_research to false for greetings, thanks, small talk, or any
message that is not asking a research question.
"""

chat_prm = """
You are ResearchPilot, a friendly research assistant.
Reply to the user conversationally and briefly. Use anything already
said in this conversation, such as the user's name, when it is relevant.
"""


def guard(state: ResearchPilotState) -> dict:
    response = invoke_structured(
        classifier_llm,
        [
            SystemMessage(content=sys_prm),
            HumanMessage(content=state["question"]),
        ],
    )

    return {"is_research": response.is_research}


def chat(state: ResearchPilotState) -> dict:
    response = llm_modl.invoke(
        [
            SystemMessage(content=chat_prm),
            *state["messages"],
        ]
    )

    return {"messages": [response], "report": response.content}
