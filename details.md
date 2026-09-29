# Interview notes: Job Application Assistant

## 30-second pitch
I built an agentic AI assistant that takes my resume and a job description, plans the work, and calls tools to score the match, tailor my resume bullets, draft a cover letter and recruiter message, and generate interview questions. Before anything reaches me it verifies every claim against my resume and revises drafts that invent experience. Nothing is sent without my approval. It runs free on Groq and Gemini, deployed on Streamlit from GitHub.

## Why is it "agentic" and not just a prompt?
A single prompt produces one answer. Here an LLM planner chooses the next tool based on the current state, observes the result, and decides again, including going back to revise when verification fails. The loop has goals, tools, state, observations and a stopping condition.

## Architecture
1. **State**: a dict holding resume, JD, requirements, match, drafts, verification results.
2. **Planner**: the LLM sees the tool list and a state summary and returns `{"thought", "tool"}` as JSON.
3. **Guardrails**: the choice is validated (known tool, inputs exist, not over caps). Invalid or failed planner output falls back to a rule-based policy.
4. **Tools**: Python functions; some call the LLM (extraction, drafting, verification), some are deterministic (scoring, skill checks).
5. **Verifier loop**: `verify_claims` flags unsupported claims; `revise_drafts` fixes them; the agent must re-verify before it may finish. Max 2 revisions.
6. **Human in the loop**: the user edits and approves; only then is the application saved to SQLite.

## Key design decisions
- **Deterministic where possible.** The match score is plain code (required skills x2, preferred x1), so it is explainable and cannot hallucinate. Skill matching uses word boundaries so "Java" does not match "JavaScript".
- **Two-layer verification.** A deterministic check catches any required skill claimed in drafts but absent from the resume, even if the LLM verifier misses it. The LLM check covers other claims (numbers, titles, achievements). "Keen to learn Kafka" is allowed; "5 years of Kafka" is not.
- **Bounded autonomy.** Step cap 14, revision cap 2, three tool errors stop the run. This prevents loops and runaway cost.
- **Graceful degradation.** If the planner fails, the agent still completes using the policy, so free-tier hiccups do not break a run.
- **Prompt-injection defence.** Resume and job-page text are wrapped as untrusted data and the prompts say to ignore instructions inside them.
- **Provider-agnostic LLM client.** Groq first, Gemini fallback, retries with backoff, skip retired models, discover live models. Learned from real 404 and 503 errors on my previous RAG project.
- **Testability.** 13 tests use a scripted fake LLM, covering hallucination removal, planner failure, invalid planner choices, revision caps, provider fallback and the tracker.

## Likely questions and answers
**How do you stop it making things up?** Grounded prompts that forbid adding facts, an explicit list of missing skills it must not claim, a two-layer verifier, a revise-and-reverify loop, and human approval.

**What if the LLM picks a bad action?** Every action is validated. Invalid choices, or a failing planner, fall back to a rule-based policy. `finish` is only accepted when all outputs exist and are verified.

**How do you prevent infinite loops or high cost?** Step and revision caps, error limits, input truncation, and an option to turn the planner off (about 8 calls instead of 15).

**Why use an LLM planner if a fixed order works?** It shows the agent pattern and lets the agent adapt, for example re-verifying after a revision or retrying a failed step. The guardrails keep it safe; the policy guarantees completion.

**How would you evaluate it?** A labelled set of job descriptions: extraction accuracy of required skills, zero unflagged unsupported claims, match-score agreement with a human, and the number of edits needed before sending.

**How would you scale it for real users?** Authentication, per-user Postgres storage instead of SQLite, async job queue for runs, tracing of each agent step (e.g. LangSmith or OpenTelemetry), rate limiting and cost budgets per user.

**Why not LangChain or LangGraph?** I wanted to show I understand the loop itself. The structure maps directly onto LangGraph (nodes are tools, edges are the policy), and migrating would be a next step.

**Privacy?** The resume goes only to the LLM provider for generation. Nothing is stored server-side unless the user approves saving, and nothing is ever sent on their behalf.

## Limitations and next steps
SQLite resets on free hosting, no OCR for scanned PDFs, English only, and a free-tier rate limit. Next: persistent storage, batch comparison across several jobs, a writing-style profile, and a LangGraph version.
