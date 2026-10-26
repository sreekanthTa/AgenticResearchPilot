import json
from typing import Literal

from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

from model import llm_modl, invoke_structured
from search import format_sources, normalize_url
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


def _cited_urls(analysis: str) -> set:
    try:
        findings = json.loads(analysis).get("findings", [])
    except Exception:
        return set()

    return {
        item.get("source_url", "")
        for item in findings
        if item.get("source_url")
    }


def verify_citations(state: ResearchPilotState) -> dict:
    results = state["search_results"]
    cited = _cited_urls(state["analysis"])
    available_urls = "\n".join(f"- {item['url']}" for item in results)

    by_norm = {normalize_url(item["url"]): item["url"] for item in results}
    wanted = {
        by_norm[normalize_url(url)]
        for url in cited
        if normalize_url(url) in by_norm
    }

    if not cited or len(wanted) < len(cited):
        evidence = format_sources(results)
        scope = "all search results"
    else:
        evidence = format_sources(results, only_urls=wanted)
        scope = "the cited sources"

    messages = [
        SystemMessage(
            content=(
                "You are a strict citation verifier for ResearchPilot. "
                "Compare every claim with the supplied source evidence. "
                "Do not assume a URL supports a claim just because it is cited. "
                "Mark claims supported only when the evidence supports them. "
                "Use partially_supported or unsupported when appropriate. "
                "A claim whose source is not in the supplied evidence is not "
                "supported - mark it partially_supported or unsupported."
            )
        ),
        HumanMessage(
            content=(
                f"Analysis:\n{state['analysis']}\n\n"
                f"Evidence from {scope}:\n{evidence}\n\n"
                f"URLs returned by search:\n{available_urls}"
            )
        ),
    ]

    response = invoke_structured(verifier_llm, messages)

    checks = [
        item.model_dump()
        for item in response.checks
    ]

    return {"citation_checks": checks}
