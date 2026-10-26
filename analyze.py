import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from model import llm_modl, invoke_structured
from search import format_sources, normalize_url
from state import ResearchPilotState


class EvidenceItem(BaseModel):
    claim: str = Field(
        description="A factual claim that supports the answer."
    )
    source_url: str = Field(
        description="The URL of the source supporting this claim."
    )
    evidence: str = Field(
        description=(
            "A SHORT verbatim quote from the supplied source content "
            "(maximum 25 words). Do not paste whole paragraphs."
        )
    )


class EvidenceAnalysis(BaseModel):
    answer: str = Field(
        description=(
            "A clear, coherent answer to the user's research question, "
            "combining relevant information from multiple sources."
        )
    )

    findings: list[EvidenceItem] = Field(
        max_length=5,
        description=(
            "The key factual claims supporting the answer, each with its "
            "source URL and a short supporting quote. Return at most 5."
        ),
    )

    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Evidence gaps, uncertainties, or limitations. Return an empty "
            "list if there are none. Always include this field."
        ),
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
3. Extract at most 5 key factual claims that support the answer.
4. For each claim, include its source URL and a SHORT verbatim quote
   (maximum 25 words) from the supplied source content.
5. Use only URLs present in the supplied search results.
6. Always fill in limitations with any agreements, disagreements,
   uncertainties, or gaps - use an empty list if there are none.
7. Do not invent facts, sources, URLs, or quotations.
8. Distinguish information explicitly supported by sources from
   broader conclusions drawn across sources.
"""


def _snap_url(cited: str, known: dict) -> str | None:
    key = normalize_url(cited)

    if key in known:
        return known[key]

    candidates = {
        url
        for norm, url in known.items()
        if norm.startswith(key) or key.startswith(norm)
    }

    if len(candidates) == 1:
        return candidates.pop()

    return None


def analyze(state: ResearchPilotState) -> dict:
    results = state["search_results"]

    messages = [
        SystemMessage(content=sys_prm),
        HumanMessage(
            content=(
                f"Research question:\n{state['question']}\n\n"
                f"Sources:\n{format_sources(results)}"
            )
        ),
    ]

    response = invoke_structured(analyzer_llm, messages)

    known = {}
    for item in results:
        known.setdefault(normalize_url(item["url"]), item["url"])

    findings = []
    dropped = 0

    for item in response.findings:
        url = _snap_url(item.source_url, known)

        if url is None:
            dropped += 1
            continue

        findings.append(
            {
                "claim": item.claim,
                "source_url": url,
                "evidence": item.evidence,
            }
        )

    limitations = list(response.limitations)

    if dropped:
        limitations.append(
            f"{dropped} claim(s) were dropped because their cited source "
            "could not be matched to any search result."
        )

    return {
        "analysis": json.dumps(
            {
                "answer": response.answer,
                "findings": findings,
                "limitations": limitations,
            }
        ),
        "revision_count": state.get("revision_count", 0) + 1,
    }
