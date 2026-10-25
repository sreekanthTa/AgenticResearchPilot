import sys
from typing import Literal

from langgraph.graph import START, END, StateGraph

from state import ResearchPilotState
from plan import plan
from search import search
from analyze import analyze
from citation import verify_citations
from report import build_report


def route_verification(state: ResearchPilotState) -> Literal["report", "analyze"]:
    checks = state.get("citation_checks", [])

    all_supported = bool(checks) and all(
        check["verdict"] == "supported"
        for check in checks
    )

    if all_supported or state.get("revision_count", 0) >= 3:
        return "report"

    return "analyze"


def build_graph():
    builder = StateGraph(ResearchPilotState)

    builder.add_node("plan", plan)
    builder.add_node("search", search)
    builder.add_node("analyze", analyze)
    builder.add_node("citation", verify_citations)
    builder.add_node("report", build_report)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "search")
    builder.add_edge("search", "analyze")
    builder.add_edge("analyze", "citation")

    builder.add_conditional_edges(
        "citation",
        route_verification,
        {
            "report": "report",
            "analyze": "analyze",
        },
    )

    builder.add_edge("report", END)

    return builder.compile()


graph = build_graph()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    result = graph.invoke({"question": "What is langchain?"})
    print("result is", result["report"])


