from typing import Literal

from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

from model import llm_modl
from state import ResearchPilotState


class CitationCheck(BaseModel):
    claim: str
    source_url: str
    verdict: Literal["supported", "partially_supported", "unsupported"]
    explanation: str


class CitationVerification(BaseModel):
    checks: list[CitationCheck]


verifier_llm = llm_modl.with_structured_output(
    CitationVerification,
    method="json_schema",
)


def verify_citations(state: ResearchPilotState) -> dict:
    messages = [
        SystemMessage(
            content=(
                "You are a strict citation verifier for ResearchPilot. "
                "Compare every claim with the supplied source evidence. "
                "Do not assume a URL supports a claim just because it is cited. "
                "Mark claims supported only when the evidence supports them. "
                "Use partially_supported or unsupported when appropriate."
            )
        ),
        HumanMessage(
            content=(
                f"Analysis:\n{state['analysis']}\n\n"
                f"Original search results:\n{state['search_results']}"
            )
        ),
    ]

    response = verifier_llm.invoke(messages)

    checks = [
        item.model_dump()
        for item in response.checks
    ]

    return {"citation_checks": checks}
