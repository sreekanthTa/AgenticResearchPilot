from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from model import llm_modl
from state import ResearchPilotState


class EvidenceItem(BaseModel):
    claim: str = Field(
        description="A factual claim that supports the answer."
    )
    source_url: str = Field(
        description="The URL of the source supporting this claim."
    )
    evidence: str = Field(
        description="The exact supporting text from the supplied source content."
    )


class EvidenceAnalysis(BaseModel):
    answer: str = Field(
        description=(
            "A clear, coherent answer to the user's research question, "
            "combining relevant information from multiple sources."
        )
    )

    findings: list[EvidenceItem] = Field(
        description=(
            "Key factual claims supporting the answer, each with its "
            "source URL and supporting evidence."
        )
    )

    limitations: list[str] = Field(
        description=(
            "Important evidence gaps, uncertainties, or limitations."
        )
    )


analyzer_llm = llm_modl.with_structured_output(
    EvidenceAnalysis,
    method="json_schema",
)


sys_prm = """
You are the evidence analyst for ResearchPilot.

1. Read the user's research question and all supplied search results.
2. Combine relevant information from multiple sources into one clear,
   coherent answer to the user's question.
3. Extract the key factual claims that support the answer.
4. For each claim, include its source URL and exact supporting text
   from the supplied source content.
5. Use only URLs present in the supplied search results.
6. Identify important agreements, disagreements, uncertainties,
   and limitations in the evidence.
7. Do not invent facts, sources, URLs, or quotations.
8. Distinguish information explicitly supported by sources from
   broader conclusions drawn across sources.
"""


def analyze(state: ResearchPilotState) -> dict:
    messages = [
        SystemMessage(content=sys_prm),
        HumanMessage(
            content=(
                f"Research question:\n{state['question']}\n\n"
                f"Search results:\n{state['search_results']}"
            )
        ),
    ]

    response = analyzer_llm.invoke(messages)

    return {
        "analysis": response.model_dump_json(),
        "revision_count": state.get("revision_count", 0) + 1,
    }
