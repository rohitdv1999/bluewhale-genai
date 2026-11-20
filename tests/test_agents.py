import pytest

from bluewhale.agents.job_match import JobMatchWorkflow, build_job_tools
from bluewhale.agents.runner import AgentRunner, FileCheckpoints
from bluewhale.llm.gateway import LLMGateway
from bluewhale.llm.types import LLMResponse, ToolCall
from tests.conftest import ScriptedProvider


def call(name, cid="c1", **args):
    return LLMResponse(content="", tool_calls=[ToolCall(cid, name, args)])


def runner(services, script, tmp_path=None, **kw):
    gateway = LLMGateway([ScriptedProvider(script)], sleep=lambda s: None)
    checkpoints = FileCheckpoints(tmp_path) if tmp_path else None
    return AgentRunner(gateway, build_job_tools(services.rag), checkpoints, **kw)


def test_tool_loop_then_final_answer(services):
    r = runner(services, [call("search_jobs", query="agentic workflows LangGraph", limit=2),
                          "Brightpath's agentic workflow role is the best fit."])
    run = r.start("Find me an agents job")
    assert run.status == "completed" and "Brightpath" in run.final_answer
    tool_event = run.events[0]
    assert tool_event["tool"] == "search_jobs" and tool_event["ok"]


def test_invalid_arguments_are_returned_to_the_model(services):
    r = runner(services, [call("search_jobs", query="x", limit=99), "done"])
    run = r.start("search")
    assert run.events[0]["ok"] is False
    assert "invalid arguments" in run.messages[3].content


def test_unknown_tool_does_not_crash(services):
    run = runner(services, [call("delete_everything"), "ok"]).start("go")
    assert run.status == "completed" and run.tool_errors == 1


def test_write_tool_pauses_for_approval_and_resumes(services, tmp_path):
    script = [call("save_shortlist", job_ids=["job_ai_agents"], note="top pick"),
              "Saved your shortlist."]
    r = runner(services, script, tmp_path)
    run = r.start("shortlist the agents job")
    assert run.status == "awaiting_approval"
    assert run.pending.name == "save_shortlist"
    assert (tmp_path / f"{run.run_id}.json").exists()

    resumed = r.resume(run.run_id, approved=True, reviewer="recruiter-1")
    assert resumed.status == "completed"
    assert any(e["type"] == "approval" and e["approved"] for e in resumed.events)
    with pytest.raises(ValueError):
        r.resume(run.run_id, approved=True, reviewer="again")


def test_rejection_is_reported_to_the_model(services):
    r = runner(services, [call("save_shortlist", job_ids=["x"], note=""), "Understood."])
    run = r.start("save")
    done = r.resume(run.run_id, approved=False, reviewer="me", note="not yet")
    assert done.status == "completed"
    assert "rejected by human reviewer" in done.messages[-2].content


def test_loop_guard_and_step_limit(services):
    same = [call("get_job", "a", job_id="job_ai_agents"), call("get_job", "b", job_id="job_ai_agents")]
    assert runner(services, same).start("loop").status == "failed"
    many = [call("get_job", str(i), job_id=f"id{i}") for i in range(10)]
    assert runner(services, many, max_steps=3, max_tool_errors=10).start("x").status == "max_steps"


def test_job_match_workflow_ranks_relevant_jobs_first(services):
    resume = (services.rag.store.documents({"doc_type": "resume"}))
    assert resume
    text = open("data/samples/resumes/resume_ai_engineer.md").read()
    state = JobMatchWorkflow(services.rag).run(text, top_n=3)
    top = [m.job_id for m in state.matches]
    assert top[0] in {"job_genai_engineer", "job_ai_agents"}
    assert state.trail[0] == "parse_resume" and state.trail[-1] == "done"
    assert state.matches[0].reasons


def test_job_match_widens_when_matches_are_weak(services):
    state = JobMatchWorkflow(services.rag, weak_threshold=0.99).run(
        "Pastry chef with 10 years of experience in French baking and team leadership." * 2)
    assert "widen_search" in state.trail
