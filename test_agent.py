"""Tests use a scripted fake LLM: fast, free, deterministic, no network."""
import json
import sys
import types

import pytest

sys.modules.setdefault("pypdf", types.SimpleNamespace(PdfReader=None))

import agent  # noqa: E402
import db  # noqa: E402
import llm as llm_mod  # noqa: E402
import tools  # noqa: E402

RESUME = (
    "Ramesh R. Software Engineer at Acme Corp (2019-2024). Built REST APIs in Java and Spring Boot. "
    "Deployed services on AWS. Wrote SQL for reporting. Led a team of 3 engineers."
)
JD = "Senior Java Developer at Globex. Required: Java, Spring Boot, Kafka, AWS. Preferred: Docker."


class FakeLLM:
    """Answers based on which tool prompt it sees. `planner` controls planner behaviour."""

    def __init__(self, planner="good", hallucinate=True):
        self.planner, self.hallucinate, self.calls, self.revised = planner, hallucinate, [], False

    def complete(self, prompt, json_mode=False):
        self.calls.append(prompt[:40])
        if "cover letter (under 250" in prompt:
            return "I built REST APIs in Java and Spring Boot at Acme Corp." + (
                " I have 5 years of Kafka experience." if self.hallucinate and not self.revised else "")
        if "LinkedIn message" in prompt:
            return "Hi, I build Java and Spring Boot APIs and would love to learn about the Globex role."
        return json.dumps(self.complete_json(prompt))

    def complete_json(self, p):
        if p.startswith("You are an agent"):
            if self.planner == "broken":
                raise llm_mod.LLMError("planner down")
            if self.planner == "silly":
                return {"thought": "done!", "tool": "finish"}
            return {"thought": "follow policy", "tool": agent.policy(self._state)}
        if "Extract from the JOB DESCRIPTION" in p:
            return {"role": "Senior Java Developer", "company": "Globex", "seniority": "senior",
                    "required_skills": ["Java", "Spring Boot", "Kafka", "AWS"], "preferred_skills": ["Docker"]}
        if "Rewrite up to 6 resume bullets" in p:
            return {"bullets": [{"original": "Built REST APIs in Java", "tailored": "Designed Java/Spring Boot REST APIs"}]}
        if "interview questions" in p:
            return {"questions": [{"type": "technical", "question": "Explain Kafka partitions", "hint": "ordering"}]}
        if "List every factual claim" in p:
            return {"unsupported": ["5 years of Kafka experience"] if "Kafka experience" in p else []}
        if "must be removed or softened" in p:
            self.revised = True
            return {"bullets": ["Designed Java/Spring Boot REST APIs"],
                    "cover_letter": "I built REST APIs in Java and Spring Boot at Acme Corp. I am keen to learn Kafka."}
        raise AssertionError("unexpected prompt: " + p[:80])


def run(fake, **kw):
    # give the fake access to live state so a "good" planner can make sensible choices
    orig = agent.plan_next

    def spy(state, llm, history):
        fake._state = state
        return orig(state, llm, history)

    agent.plan_next = spy
    try:
        return agent.run_agent(RESUME, JD, fake, **kw)
    finally:
        agent.plan_next = orig


def test_skill_matching_is_word_boundary_safe():
    assert tools.skill_in_text("Java", "Java and Spring")
    assert not tools.skill_in_text("Java", "JavaScript only")
    assert tools.skill_in_text("C++", "Wrote C++ code")
    assert tools.skill_in_text("Node.js", "node.js services")


def test_score_match_weights_required_double():
    s = {"resume": RESUME, "requirements": {"required_skills": ["Java", "Kafka"], "preferred_skills": ["Docker"]}}
    tools.score_match(s)
    assert s["match"]["score"] == 40  # 2 of 5 points
    assert s["match"]["missing_required"] == ["Kafka"]


def test_agent_completes_and_verifier_removes_hallucination():
    fake = FakeLLM()
    r = run(fake)
    s = r.state
    assert r.stopped_reason == "completed"
    assert s["verified"] is True and s["revisions"] == 1
    assert "5 years of Kafka" not in s["cover_letter"]
    tools_used = [st.tool for st in r.steps]
    assert tools_used.index("revise_drafts") > tools_used.index("verify_claims")
    assert tools_used.count("verify_claims") == 2  # re-verified after revising


def test_deterministic_check_catches_claimed_missing_skill_even_if_llm_says_ok():
    s = {"resume": RESUME, "requirements": {"required_skills": ["Kafka"], "preferred_skills": []},
         "bullets": [{"tailored": "Built Kafka pipelines"}], "cover_letter": "", "outreach": ""}

    class OkLLM:
        def complete_json(self, p):
            return {"unsupported": []}

    tools.verify_claims(s, OkLLM())
    assert s["verified"] is False and "Kafka" in s["unsupported"]


def test_keen_to_learn_is_not_flagged():
    s = {"resume": RESUME, "requirements": {"required_skills": ["Kafka"], "preferred_skills": []},
         "bullets": [], "cover_letter": "I am keen to learn Kafka.", "outreach": ""}

    class OkLLM:
        def complete_json(self, p):
            return {"unsupported": []}

    tools.verify_claims(s, OkLLM())
    assert s["verified"] is True


def test_invalid_planner_choice_is_overridden_by_policy():
    r = run(FakeLLM(planner="silly"))
    assert r.stopped_reason == "completed"
    assert r.steps[0].tool == "extract_requirements" and r.steps[0].chosen_by == "policy"


def test_planner_failure_falls_back_to_policy():
    r = run(FakeLLM(planner="broken"))
    assert r.stopped_reason == "completed"
    assert all(s.chosen_by == "policy" for s in r.steps)


def test_revision_cap_stops_endless_loop():
    class Stubborn(FakeLLM):
        def complete_json(self, p):
            if "List every factual claim" in p:
                return {"unsupported": ["made-up award"]}
            return super().complete_json(p)

    r = run(Stubborn(hallucinate=False))
    assert r.stopped_reason == "completed"
    assert r.state["revisions"] == agent.MAX_REVISIONS and r.state["verified"] is False
    assert len(r.steps) <= agent.MAX_STEPS


def test_tool_errors_stop_the_run():
    class Down(FakeLLM):
        def complete_json(self, p):
            if p.startswith("You are an agent"):
                return super().complete_json(p)
            raise llm_mod.LLMError("provider down")

    r = run(Down())
    assert r.stopped_reason == "too many tool errors"


def test_llm_falls_back_on_retired_model_and_retries_overload(monkeypatch):
    calls = []

    def groq(key, model, prompt, j):
        calls.append(model)
        if model == "llama-3.3-70b-versatile":
            raise llm_mod.LLMError("Groq 503: over capacity")
        raise llm_mod.LLMError("Groq 404: model_not_found")

    def gem(key, model, prompt, j):
        calls.append(model)
        return '```json\n{"ok": true}\n```'

    monkeypatch.setattr(llm_mod, "_groq_call", groq)
    monkeypatch.setattr(llm_mod, "_gemini_call", gem)
    monkeypatch.setattr(llm_mod, "_discover_groq", lambda k: [])
    m = llm_mod.LLM("g", "k", sleep=lambda s: None)
    assert m.complete_json("x") == {"ok": True}
    assert calls[:3] == ["llama-3.3-70b-versatile"] * 3  # retried transient error 3 times
    assert "llama-3.1-8b-instant" in calls  # then moved on
    assert m.used.startswith("gemini:")


def test_all_models_failing_raises(monkeypatch):
    monkeypatch.setattr(llm_mod, "_gemini_call", lambda *a: (_ for _ in ()).throw(llm_mod.LLMError("404")))
    monkeypatch.setattr(llm_mod, "_discover_gemini", lambda k: [])
    with pytest.raises(llm_mod.LLMError):
        llm_mod.LLM(None, "k", sleep=lambda s: None).complete("x")


def test_parse_json_tolerates_prose():
    assert llm_mod.parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}


def test_tracker_roundtrip(tmp_path):
    con = db.connect(str(tmp_path / "t.db"))
    i = db.save(con, {"requirements": {"company": "Globex", "role": "Dev"}, "match": {"score": 80}})
    db.update_status(con, i, "Applied")
    row = db.list_all(con)[0]
    assert (row["company"], row["score"], row["status"]) == ("Globex", 80, "Applied")
    with pytest.raises(ValueError):
        db.update_status(con, i, "Hacked")
