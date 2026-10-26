from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm_modl = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=2048,
    reasoning_effort="low",
)


def invoke_structured(llm, messages, attempts: int = 3):
    for attempt in range(attempts):
        try:
            return llm.invoke(messages)
        except Exception:
            if attempt == attempts - 1:
                raise
