"""Streamlit UI: upload documents, ask questions, inspect the retrieval.

There is no login screen. The API still requires a bearer token -- the ACL is
real and enforced in the data layer -- but for local single-user operation the
UI provisions one account on first run and caches the token, because asking a
developer to register before they can try their own tool is friction with no
security benefit on localhost.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import httpx
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / ".local_token"
TOKEN_CACHE = Path(os.getenv("TOKEN_CACHE", DEFAULT_CACHE))

st.set_page_config(page_title="Enterprise RAG Assistant", page_icon=":books:", layout="wide")


@st.cache_resource(show_spinner="Starting up...")
def local_token() -> str:
    """Reuse the cached token, or register a throwaway local account."""
    if TOKEN_CACHE.exists():
        cached = TOKEN_CACHE.read_text(encoding="utf-8").strip()
        if cached:
            return cached
    email = f"local-{uuid.uuid4().hex[:12]}@localhost"
    r = httpx.post(f"{API}/auth/register", json={"email": email}, timeout=60)
    r.raise_for_status()
    token = r.json()["token"]
    TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_CACHE.write_text(token, encoding="utf-8")
    return token


def api(method: str, path: str, **kw) -> httpx.Response:
    headers = {"Authorization": f"Bearer {local_token()}"}
    return httpx.request(method, f"{API}{path}", headers=headers, timeout=300, **kw)


st.title("Enterprise RAG Assistant")
st.caption("Answers are grounded in your documents, with a page citation for every claim.")

try:
    token = local_token()
except httpx.HTTPError:
    st.error(f"Cannot reach the API at {API}. Start it with: uvicorn app.api.main:app --reload")
    st.stop()

# -- sidebar: documents ----------------------------------------------------

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload a PDF, TXT or MD", type=["pdf", "txt", "md"])
    if uploaded is not None and st.button("Index document", width="stretch"):
        with st.spinner(f"Indexing {uploaded.name}..."):
            resp = api(
                "POST",
                "/documents",
                files={"file": (uploaded.name, uploaded.getvalue())},
            )
        if resp.status_code == 201:
            body = resp.json()
            st.success(f"Indexed {body['pages']} pages into {body['chunks']} chunks.")
        else:
            st.error(resp.text)

    docs_resp = api("GET", "/documents")
    docs = docs_resp.json().get("documents", []) if docs_resp.status_code == 200 else []

    selected: list[str] = []
    if docs:
        labels = {f"{d['title']} ({d['pages']}p)": d["doc_id"] for d in docs}
        chosen = st.multiselect("Restrict the search to", list(labels))
        selected = [labels[c] for c in chosen]
        st.caption(f"{len(docs)} document(s) indexed")
    else:
        st.info("Upload a document to get started.")

# -- main: ask -------------------------------------------------------------

question = st.text_input("Ask a question", placeholder="What is the notice period for termination?")

if st.button("Ask", type="primary", disabled=not question):
    with st.spinner("Retrieving and answering..."):
        resp = api("POST", "/ask", json={"question": question, "doc_ids": selected or None})

    if resp.status_code == 502:
        st.error(resp.json().get("detail", resp.text))
    elif resp.status_code != 200:
        st.error(f"{resp.status_code}: {resp.text}")
    else:
        body = resp.json()
        if body["refused"]:
            st.warning(body["answer"])
            st.caption(f"Reason: {body['reason']}")
        else:
            st.markdown(body["answer"])
            if body["citations"]:
                st.subheader("Sources")
                for c in body["citations"]:
                    st.markdown(f"**[{c['marker']}]** {c['title']} — page {c['page']}")

        with st.expander(f"Retrieved passages ({len(body['passages'])})"):
            for i, p in enumerate(body["passages"], start=1):
                st.markdown(f"**[{i}]** {p['title']}, page {p['page']} — score {p['score']}")
                st.text(p["text"][:1200])
                st.divider()
