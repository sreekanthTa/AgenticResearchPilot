import time

from ddgs import DDGS

from state import ResearchPilotState

MAX_PER_TASK = 3
MAX_SOURCES = 12
MAX_SNIPPET = 300
MAX_TITLE = 100


def _search_task(
    query: str, max_results: int = MAX_PER_TASK, retries: int = 3
) -> list[dict]:
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                return ddgs.text(query, max_results=max_results)
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(2 * (attempt + 1))

    return []


def _clean(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def normalize_url(url: str) -> str:
    return url.split("?")[0].split("#")[0].rstrip("/").lower()


def search(state: ResearchPilotState) -> dict:
    all_results = []
    seen_urls = set()

    for task in state["tasks"]:
        for item in _search_task(task):
            url = item.get("href", "")

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            all_results.append(
                {
                    "task": task,
                    "title": _clean(item.get("title", ""), MAX_TITLE),
                    "url": url,
                    "content": _clean(item.get("body", ""), MAX_SNIPPET),
                }
            )

            if len(all_results) >= MAX_SOURCES:
                return {"search_results": all_results}

    return {"search_results": all_results}


def format_sources(results, only_urls=None) -> str:
    lines = []

    for item in results:
        if only_urls is not None and item["url"] not in only_urls:
            continue

        lines.append(f"[{len(lines) + 1}] {item['url']}\n{item['content']}")

    return "\n\n".join(lines)
