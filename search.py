import time

from ddgs import DDGS

from state import ResearchPilotState


def _search_task(query: str, max_results: int = 5, retries: int = 3) -> list[dict]:
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                return ddgs.text(query, max_results=max_results)
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(2 * (attempt + 1))

    return []


def search(state: ResearchPilotState) -> dict:
    all_results = []

    for task in state["tasks"]:
        for item in _search_task(task):
            all_results.append(
                {
                    "task": task,
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "content": item.get("body", ""),
                }
            )

    return {"search_results": all_results}
