# Job Application Assistant (Agentic AI)

**Live demo: LIVE_URL**

Detailed notes for this project: [details.md](https://github.com/Ramesh-Ramasamy/job-application-assistant/blob/main/details.md)

An agentic AI app: give it your resume and a job description, and it **plans**, **calls tools**, **verifies its own drafts against your resume**, and **waits for your approval**. It never sends or submits anything on its own.

## What it produces
- Match score with matched and missing skills
- Tailored resume bullets (only rephrasing facts already in your resume)
- Cover letter and a short recruiter/referral message, in your chosen tone
- Role-specific interview questions with answer hints
- A tracker of approved applications with status (Drafted, Applied, Interviewing, ...)

## How the agent works

```
goal -> [planner picks next tool] -> act -> observe -> repeat -> verify -> revise if needed -> finish -> human approval
```

| Tool | What it does |
|---|---|
| `extract_requirements` | LLM extracts role, company, required/preferred skills (JSON) |
| `score_match` | Deterministic, explainable score: required skills x2, preferred x1 |
| `tailor_bullets` | Rewrites bullets using only resume facts |
| `draft_cover_letter` / `draft_outreach` | Drafts in the chosen tone, never claiming missing skills |
| `generate_interview_questions` | Technical questions (including your gaps) and behavioural ones |
| `verify_claims` | Two layers: a deterministic skill check plus an LLM claim check |
| `revise_drafts` | Removes unsupported claims, then the agent re-verifies |

**Guardrails:** only known tools, only when their inputs exist; step cap (14) and revision cap (2); `finish` is accepted only when all outputs exist and are verified; if the LLM planner returns something invalid or fails, a rule-based policy picks the next step. Job-page text and resume text are treated as untrusted data, never as instructions.

**LLM layer:** Groq Llama first, Gemini as fallback; transient errors (429/503) are retried with exponential backoff, retired models are skipped, and live models are discovered from each provider's API.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, approval flow, tracker |
| `agent.py` | Plan-act-observe loop and guardrails |
| `tools.py` | The tools and the verifier |
| `llm.py` | Provider-agnostic LLM client with retries and fallbacks |
| `db.py` | SQLite tracker |
| `test_agent.py` | 13 tests with a scripted fake LLM (no network, no cost) |

## Run locally
```bash
pip install -r requirements.txt
export GROQ_API_KEY=...     # free: https://console.groq.com/keys
export GEMINI_API_KEY=...   # optional fallback, free: https://aistudio.google.com/apikey
streamlit run app.py
```

## Deploy free (Streamlit Community Cloud)
Create an app from this repo (branch `main`, file `app.py`) and add the keys in **Advanced settings > Secrets**:
```toml
GROQ_API_KEY = "..."
GEMINI_API_KEY = "..."
```

## Tests
```bash
pip install -r requirements-dev.txt
pytest -q
```

## Limitations
- The tracker uses SQLite; on free Streamlit hosting it resets when the app restarts (download your drafts).
- Scanned PDFs need pasted text (no OCR).
- Free-tier rate limits apply; a run makes roughly 15 LLM calls with the planner on, about 8 with it off.
