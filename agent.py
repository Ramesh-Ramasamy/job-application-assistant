"""The agent loop: plan -> act (call a tool) -> observe -> repeat, with guardrails.

The LLM chooses the next tool. Guardrails keep it safe and bounded:
  * only known tools, only when their inputs exist (preconditions)
  * a step cap and a revision cap (no infinite loops / runaway cost)
  * 'finish' is only accepted once every output exists and drafts are verified (or revisions are exhausted)
  * if the planner returns something invalid, a rule-based policy picks the next step instead
The agent never sends or submits anything: it only drafts. Saving happens after the user approves in the UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm import LLM
from tools import TOOLS

MAX_STEPS = 14
MAX_REVISIONS = 2
OUTPUTS = ["requirements", "match", "bullets", "cover_letter", "outreach", "questions"]


@dataclass
class Step:
    n: int
    thought: str
    tool: str
    observation: str
    chosen_by: str  # "planner" or "policy"


@dataclass
class Result:
    state: dict
    steps: list[Step] = field(default_factory=list)
    stopped_reason: str = ""


def _has(state: dict, key: str) -> bool:
    v = state.get(key)
    return v is not None and v != [] and v != ""


def can_finish(state: dict) -> bool:
    done = all(_has(state, k) for k in OUTPUTS) and "verified" in state
    return done and (state["verified"] or state.get("revisions", 0) >= MAX_REVISIONS)


def allowed(state: dict, tool: str) -> bool:
    if tool == "finish":
        return can_finish(state)
    if tool not in TOOLS:
        return False
    _, pre, _ = TOOLS[tool]
    if not all(_has(state, p) for p in pre):
        return False
    if tool == "revise_drafts" and (state.get("verified", True) or state.get("revisions", 0) >= MAX_REVISIONS):
        return False
    if tool == "verify_claims" and state.get("verified") is not None and not state.get("needs_reverify"):
        return False
    return True


def policy(state: dict) -> str:
    """Rule-based fallback: the sensible next step given the current state."""
    for key, tool in [("requirements", "extract_requirements"), ("match", "score_match"),
                      ("bullets", "tailor_bullets"), ("cover_letter", "draft_cover_letter"),
                      ("outreach", "draft_outreach"), ("questions", "generate_interview_questions")]:
        if not _has(state, key):
            return tool
    if "verified" not in state or state.get("needs_reverify"):
        return "verify_claims"
    if not state["verified"] and state.get("revisions", 0) < MAX_REVISIONS:
        return "revise_drafts"
    return "finish"


def _status(state: dict) -> str:
    rows = [f"- {k}: {'done' if _has(state, k) else 'missing'}" for k in OUTPUTS]
    rows.append(f"- verified: {state.get('verified', 'not checked')}; revisions: {state.get('revisions', 0)}")
    if state.get("unsupported"):
        rows.append(f"- unsupported claims: {state['unsupported'][:5]}")
    return "\n".join(rows)


def plan_next(state: dict, llm: LLM, history: list[Step]) -> tuple[str, str]:
    tools = "\n".join(f"- {name}: {desc}" for name, (_, _, desc) in TOOLS.items())
    recent = "\n".join(f"{s.n}. {s.tool} -> {s.observation}" for s in history[-6:]) or "(none yet)"
    out = llm.complete_json(
        "You are an agent preparing a job application. Goal: produce requirements, match score, tailored bullets, "
        "cover letter, outreach message and interview questions, then verify all drafts are supported by the resume, "
        "revising if not. Choose exactly ONE next action.\n"
        f"Tools:\n{tools}\n- finish: stop when everything is done and verified\n\n"
        f"Current state:\n{_status(state)}\n\nRecent steps:\n{recent}\n\n"
        'Return JSON: {"thought": "<one short sentence>", "tool": "<tool name>"}'
    )
    return str(out.get("tool", "")).strip(), str(out.get("thought", "")).strip()


def run_agent(resume: str, jd: str, llm: LLM, tone: str = "professional", use_planner: bool = True,
              on_step=None) -> Result:
    state: dict = {"resume": resume, "jd": jd, "tone": tone}
    res = Result(state=state)
    for n in range(1, MAX_STEPS + 1):
        tool, thought, chosen_by = "", "", "policy"
        if use_planner:
            try:
                tool, thought = plan_next(state, llm, res.steps)
                chosen_by = "planner"
            except Exception as e:  # noqa: BLE001  planner failure should not kill the run
                thought = f"planner unavailable ({str(e)[:80]}), using policy"
        if not allowed(state, tool):
            if chosen_by == "planner" and tool:
                thought = f"planner chose '{tool}' which is not valid now; policy overrides"
            tool, chosen_by = policy(state), "policy"
            thought = thought or "next step by policy"
        if tool == "finish":
            res.steps.append(Step(n, thought, "finish", "done", chosen_by))
            res.stopped_reason = "completed"
            break
        fn = TOOLS[tool][0]
        try:
            obs = fn(state, llm)
            if tool == "revise_drafts":
                state["needs_reverify"] = True
            if tool == "verify_claims":
                state["needs_reverify"] = False
        except Exception as e:  # noqa: BLE001
            obs = f"ERROR: {str(e)[:200]}"
            state.setdefault("errors", []).append(f"{tool}: {e}")
            if len(state["errors"]) >= 3:
                res.steps.append(Step(n, thought, tool, obs, chosen_by))
                res.stopped_reason = "too many tool errors"
                break
        step = Step(n, thought, tool, obs, chosen_by)
        res.steps.append(step)
        if on_step:
            on_step(step)
    else:
        res.stopped_reason = "step limit reached"
    return res
