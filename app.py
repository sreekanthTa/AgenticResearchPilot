import streamlit as st
from langchain_core.messages import HumanMessage

import store

st.set_page_config(page_title="ResearchPilot", page_icon="🔎")


@st.cache_resource
def setup():
    store.init_db()


@st.cache_resource
def get_graph():
    from graph import build_graph

    return build_graph(store.build_checkpointer())


setup()
graph = get_graph()

THEMES = ["Light", "Dark"]


def apply_theme(theme: str) -> None:
    dark = theme == "Dark"

    if dark:
        background = "#0e1117"
        panel = "#262730"
        text = "#fafafa"
        scheme = "dark"
    else:
        background = "#ffffff"
        panel = "#f0f2f6"
        text = "#31333f"
        scheme = "light"

    st.markdown(
        f"""
        <style>
        :root, .stApp {{
            --background-color: {background};
            --secondary-background-color: {panel};
            --text-color: {text};
            color-scheme: {scheme};
        }}
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {{
            background-color: {background};
            color: {text};
        }}
        [data-testid="stSidebar"] {{
            background-color: {panel};
        }}
        [data-testid="stSidebar"] *,
        .stMarkdown,
        .stMarkdown p,
        h1, h2, h3, h4, h5, h6 {{
            color: {text};
        }}
        [data-testid="stChatMessage"] {{
            background-color: {panel};
        }}
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] > div {{
            background-color: {panel};
            color: {text};
        }}
        textarea, input {{
            background-color: {panel};
            color: {text};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_auth():
    st.title("ResearchPilot")
    st.caption("Sign in or create an account to keep your research.")

    sign_in, sign_up = st.tabs(["Sign in", "Create account"])

    with sign_in:
        email = st.text_input("Email", key="signin_email")
        password = st.text_input("Password", type="password", key="signin_password")
        if st.button("Sign in", key="signin_button"):
            row = store.get_user(email)
            if row and store.verify_password(password, row[1]):
                st.session_state.user_id = str(row[0])
                st.session_state.email = email.lower().strip()
                st.rerun()
            else:
                st.error("Invalid email or password.")

    with sign_up:
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input(
            "Password", type="password", key="signup_password"
        )
        if st.button("Create account", key="signup_button"):
            if len(new_password) < 8:
                st.error("Password must be at least 8 characters.")
            elif store.get_user(new_email):
                st.error("That email is already registered.")
            else:
                user_id = store.create_user(new_email, new_password)
                st.session_state.user_id = user_id
                st.session_state.email = new_email.lower().strip()
                st.rerun()


def load_history(thread_id):
    state = graph.get_state({"configurable": {"thread_id": thread_id}}).values

    pairs = []
    for message in state.get("messages", []):
        role = "user" if message.type == "human" else "assistant"
        pairs.append((role, message.content))

    return pairs


if not st.session_state.get("user_id"):
    render_auth()
    st.stop()

user_id = st.session_state["user_id"]
theme = store.get_pref(user_id, "theme", "Light")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Account")
    st.caption(st.session_state.get("email", ""))

    if st.button("Log out", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.divider()

    chosen = st.radio("Theme", THEMES, index=THEMES.index(theme), horizontal=True)
    if chosen != theme:
        store.set_pref(user_id, "theme", chosen)
        theme = chosen

    st.divider()

    if st.button("+ New conversation", use_container_width=True):
        st.session_state.thread_id = None
        st.session_state.history = []
        st.rerun()

    st.caption("Your conversations")
    for thread_id, title, _ in store.list_threads(user_id):
        if st.button(title[:38], key=f"thread-{thread_id}", use_container_width=True):
            st.session_state.thread_id = thread_id
            st.session_state.history = load_history(thread_id)
            st.rerun()

apply_theme(theme)

st.title("ResearchPilot")

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

prompt = st.chat_input("Ask a research question, or just say hi...")

if prompt:
    if st.session_state.thread_id is None:
        st.session_state.thread_id = store.create_thread(user_id, prompt[:60])

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
