import json

from langchain_core.messages import AIMessage

from state import ResearchPilotState


def build_report(state: ResearchPilotState) -> dict:
    analysis = json.loads(state["analysis"])
    checks = state.get("citation_checks", [])

    lines = [analysis["answer"], ""]

    if analysis.get("findings"):
        lines.append("## Supporting evidence")
        for item in analysis["findings"]:
            lines.append(f"- {item['claim']} ({item['source_url']})")

    problems = [check for check in checks if check["verdict"] != "supported"]
    if problems:
        lines.append("\n## Unverified claims")
        for check in problems:
            lines.append(f"- [{check['verdict']}] {check['claim']} ({check['source_url']})")

    if analysis.get("limitations"):
        lines.append("\n## Limitations")
        for limitation in analysis["limitations"]:
            lines.append(f"- {limitation}")

    text = "\n".join(lines)

    return {"report": text, "messages": [AIMessage(content=text)]}
