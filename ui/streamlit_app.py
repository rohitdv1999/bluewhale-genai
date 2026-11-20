"""Streamlit front end for demos and internal testing.

    streamlit run ui/streamlit_app.py      # expects the API on BW_API_URL
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API = os.environ.get("BW_API_URL", "http://localhost:8000")
HEADERS = {"x-api-key": os.environ["BW_API_KEY"]} if os.environ.get("BW_API_KEY") else {}

st.set_page_config(page_title="BlueWhale GenAI", layout="wide")
st.title("BlueWhale GenAI")
ask_tab, match_tab, agent_tab, ingest_tab = st.tabs(["Ask", "Match", "Agent", "Add document"])


def post(path: str, payload: dict) -> dict:
    r = requests.post(f"{API}{path}", json=payload, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


with ask_tab:
    question = st.text_input("Question", "Which job needs MLflow experiment tracking?")
    doc_type = st.selectbox("Search in", ["all", "job_description", "resume"])
    if st.button("Ask", type="primary"):
        payload = {"question": question}
        if doc_type != "all":
            payload["doc_type"] = doc_type
        st.session_state["answer"] = post("/v1/ask", payload)
    data = st.session_state.get("answer")
    if data:
        (st.warning if data["refused"] else st.success)(data["answer"])
        for c in data["citations"]:
            st.caption(f"[{c['n']}] {c['title']} - {c['section']} (score {c['score']:.3f})")
        with st.expander("Usage and trace"):
            st.json({"trace_id": data["trace_id"], "grounding": data["grounding_score"],
                     **data["usage"]})
        col1, col2 = st.columns(2)
        if col1.button("Helpful"):
            post("/v1/feedback", {"trace_id": data["trace_id"], "helpful": True})
        if col2.button("Not helpful"):
            post("/v1/feedback", {"trace_id": data["trace_id"], "helpful": False})

with match_tab:
    resume = st.text_area("Paste a resume", height=260)
    if st.button("Find matching jobs") and resume:
        data = post("/v1/match", {"resume_text": resume, "top_n": 5})
        st.write("Detected skills:", ", ".join(data["candidate"]["skills"]))
        for m in data["matches"]:
            st.subheader(f"{m['title']}  -  {m['score']:.0%}")
            for reason in m["reasons"]:
                st.write("- " + reason)

with agent_tab:
    goal = st.text_input("Goal", "Find agentic AI roles and shortlist the best one")
    if st.button("Run agent"):
        st.session_state["run"] = post("/v1/agent/runs", {"goal": goal})
    run = st.session_state.get("run")
    if run:
        st.write(f"Status: **{run['status']}**  -  steps {run['steps']}")
        if run["final_answer"]:
            st.info(run["final_answer"])
        if run["pending_action"]:
            st.warning(f"Needs approval: {run['pending_action']}")
            c1, c2 = st.columns(2)
            for col, ok in ((c1, True), (c2, False)):
                if col.button("Approve" if ok else "Reject"):
                    st.session_state["run"] = post(f"/v1/agent/runs/{run['run_id']}/review",
                                                   {"approved": ok, "reviewer": "ui-user"})
                    st.rerun()
        st.json(run["events"])

with ingest_tab:
    text = st.text_area("Resume or job description text", height=260, key="ingest")
    kind = st.selectbox("Type", ["auto", "resume", "job_description"])
    if st.button("Index") and text:
        payload = {"text": text}
        if kind != "auto":
            payload["doc_type"] = kind
        st.json(post("/v1/documents", payload))
