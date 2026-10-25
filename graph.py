import sys
from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from state import ResearchPilotState
from guard import guard, chat
from plan import plan
from search import search
from analyze import analyze
from citation import verify_citations
from report import build_report

checkpointer = InMemorySaver()


def route_intent(state: ResearchPilotState) -> Literal["plan", "chat"]:
    return "plan" if state.get("is_research") else "chat"


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

    builder.add_node("guard", guard)
    builder.add_node("chat", chat)
    builder.add_node("plan", plan)
    builder.add_node("search", search)
    builder.add_node("analyze", analyze)
    builder.add_node("citation", verify_citations)
    builder.add_node("report", build_report)

    builder.add_edge(START, "guard")

    builder.add_conditional_edges(
        "guard",
        route_intent,
        {
            "plan": "plan",
            "chat": "chat",
        },
    )

    builder.add_edge("chat", END)
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

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


config = {"configurable": {"thread_id": "sreekanth-1"}}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    print("ResearchPilot - type a question, or 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            break

        result = graph.invoke(
            {"question": question, "messages": [HumanMessage(content=question)]},
            config,
        )
        print(f"\nResearchPilot: {result['report']}\n")
