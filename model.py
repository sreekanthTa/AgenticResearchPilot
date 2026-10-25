from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm_modl = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=1024,
    reasoning_effort="low",
)
