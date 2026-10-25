import uuid

import streamlit as st
from langchain_core.messages import HumanMessage


@st.cache_resource
def get_graph():
    from graph import graph

    return graph


st.set_page_config(page_title="ResearchPilot", page_icon="🔎")

graph = get_graph()

st.title("ResearchPilot")
st.caption("Ask a research question, or just say hi.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Session")
    st.caption("Thread id")
    st.code(st.session_state.thread_id)

    if st.button("New chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

prompt = st.chat_input("Ask a research question...")

if prompt:
    st.session_state.history.append(("user", prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = graph.invoke(
                    {
                        "question": prompt,
                        "messages": [HumanMessage(content=prompt)],
                    },
                    config,
                )
            answer = result["report"]
        except Exception as exc:
            answer = f"Something went wrong: {exc}"

        st.markdown(answer)

    st.session_state.history.append(("assistant", answer))
