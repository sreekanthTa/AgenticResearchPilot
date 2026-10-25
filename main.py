import sys
import uuid

from langchain_core.messages import HumanMessage

from graph import graph


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

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


if __name__ == "__main__":
    main()
