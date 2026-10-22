"""Minimal chat page for support agents. Run: streamlit run src/grounded_rag/ui/streamlit_app.py"""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
HEADERS = {"X-API-Key": key} if (key := os.environ.get("API_KEY")) else {}
REQUEST_TIMEOUT_SECONDS = 60.0
EXAMPLE_QUESTIONS = (
    "How do I factory reset the S7?",
    "How long is the battery warranty?",
    "What does error E7 mean?",
    "What is the CEO's salary?",
)


def ask(question: str) -> dict:
    try:
        response = httpx.post(
            f"{API_URL}/chat",
            json={"question": question},
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError:
        return {"error": "Can't reach the answer service. Is the API running?"}
    if response.status_code == 401:
        return {"error": "The chat page is not authorised to call the API (check API_KEY)."}
    if response.status_code == 422:
        return {"error": "Please enter a question (up to 2,000 characters)."}
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", "Unknown error")
        except ValueError:
            detail = f"HTTP {response.status_code}"
        return {"error": f"The service is temporarily unavailable: {detail}"}
    return response.json()


def library_size() -> int | None:
    try:
        return httpx.get(f"{API_URL}/health", timeout=5.0).json()["chunks"]
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def render_reply(reply: dict) -> None:
    if "error" in reply:
        st.error(reply["error"])
        return
    if reply["refused"]:
        st.warning(reply["answer"])
        return
    st.markdown(reply["answer"])
    for number, source in enumerate(reply["sources"], start=1):
        label = f"[{number}] {source['doc_name']} — page {source['page']}"
        with st.expander(label):
            st.caption(f"Relevance {source['score']:.2f}")
            st.write(source["excerpt"])


def sidebar() -> str | None:
    with st.sidebar:
        st.header("Document library")
        size = library_size()
        if size is None:
            st.error("API unreachable")
        else:
            st.metric("Indexed passages", size)
        st.header("Try asking")
        # Render every button before deciding, so none disappear on the rerun after a click.
        clicks = [st.button(q, use_container_width=True) for q in EXAMPLE_QUESTIONS]
        if st.button("Clear chat", type="secondary"):
            st.session_state.history = []
    return next((q for q, clicked in zip(EXAMPLE_QUESTIONS, clicks, strict=True) if clicked), None)


def main() -> None:
    st.set_page_config(page_title="Support Docs Assistant", page_icon="📄")
    st.title("📄 Support Docs Assistant")
    st.caption("Answers come only from company documents, with page citations.")
    st.session_state.setdefault("history", [])

    clicked = sidebar()
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            render_reply(turn["reply"])

    question = st.chat_input("Ask about products, policies or procedures…") or clicked
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"), st.spinner("Searching documents…"):
            reply = ask(question)
            render_reply(reply)
        st.session_state.history = [
            *st.session_state.history,
            {"question": question, "reply": reply},
        ]


main()
