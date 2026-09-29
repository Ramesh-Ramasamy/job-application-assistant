import os

import streamlit as st

import db
from agent import MAX_STEPS, run_agent
from llm import LLM, LLMError
from tools import fetch_job_page, pdf_to_text

BUILD = "2026-09-29-a"

st.set_page_config(page_title="Job Application Assistant", page_icon="🧭", layout="wide")
st.title("🧭 Job Application Assistant")
st.caption("An agentic AI that plans, calls tools, verifies its own drafts against your resume, and waits for your approval.")


def secret(name: str) -> str | None:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


groq_key, gemini_key = secret("GROQ_API_KEY"), secret("GEMINI_API_KEY")

with st.sidebar:
    st.header("Settings")
    if not (groq_key or gemini_key):
        groq_key = st.text_input("Groq API key", type="password") or None
    tone = st.selectbox("Tone", ["professional", "warm", "concise", "enthusiastic"])
    use_planner = st.toggle("LLM planner chooses each step", value=True,
                            help="Off = a fixed rule-based order. On = the model decides the next tool (with guardrails).")
    st.caption(f"Build {BUILD} | providers: {'Groq ' if groq_key else ''}{'Gemini' if gemini_key else ''}")
    st.caption("Privacy: your resume is sent only to the LLM provider to generate drafts. Nothing is sent or submitted on your behalf.")

if "con" not in st.session_state:
    st.session_state.con = db.connect()

tab_run, tab_tracker = st.tabs(["Prepare application", "Tracker"])

with tab_run:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Your resume")
        pdf = st.file_uploader("Upload resume PDF", type="pdf")
        resume_text = st.text_area("...or paste resume text", height=160)
    with c2:
        st.subheader("2. The job")
        jd_url = st.text_input("Job posting URL (optional, public pages only)")
        jd_text = st.text_area("...or paste the job description", height=160)

    if st.button("Run agent", type="primary"):
        resume = pdf_to_text(pdf.getvalue()) if pdf else resume_text.strip()
        jd = jd_text.strip()
        if not jd and jd_url:
            try:
                jd = fetch_job_page(jd_url)
            except Exception as e:
                st.error(f"Could not fetch the page ({e}). Paste the job description instead.")
        if not resume or not jd:
            st.warning("Please provide both a resume and a job description.")
            st.stop()
        if len(resume) < 200:
            st.warning("The resume text looks very short. If it is a scanned PDF, paste the text instead.")
        try:
            llm = LLM(groq_key, gemini_key)
        except LLMError as e:
            st.error(str(e))
            st.stop()
        with st.status("Agent working...", expanded=True) as box:
            def show(step):
                box.write(f"**{step.n}. {step.tool}** ({step.chosen_by}) — {step.observation}")
            result = run_agent(resume, jd, llm, tone=tone, use_planner=use_planner, on_step=show)
            box.update(label=f"Agent {result.stopped_reason} in {len(result.steps)} steps (max {MAX_STEPS})",
                       state="complete" if result.stopped_reason == "completed" else "error")
        st.session_state.result = result
        st.session_state.jd_url = jd_url

    result = st.session_state.get("result")
    if result:
        s = result.state
        if s.get("errors"):
            st.error("Some steps failed: " + " | ".join(s["errors"][-3:]))
        if s.get("unsupported"):
            st.warning("Still unverified after revisions, please edit these before using: " + "; ".join(s["unsupported"]))
        elif s.get("verified"):
            st.success("All drafted claims were verified against your resume.")

        m, req = s.get("match", {}), s.get("requirements", {})
        t = st.tabs(["Match", "Resume bullets", "Cover letter", "Outreach", "Interview prep", "Agent trace"])
        with t[0]:
            st.metric(f"Match for {req.get('role') or 'role'} {('at ' + req['company']) if req.get('company') else ''}",
                      f"{m.get('score', 0)}%")
            st.write("**Matched:**", ", ".join(m.get("matched", [])) or "none")
            st.write("**Missing (required):**", ", ".join(m.get("missing_required", [])) or "none")
            st.write("**Missing (preferred):**", ", ".join(m.get("missing_preferred", [])) or "none")
        with t[1]:
            for i, b in enumerate(s.get("bullets", [])):
                st.caption(f"Original: {b.get('original', '')}")
                b["tailored"] = st.text_area(f"Bullet {i + 1}", b["tailored"], key=f"b{i}")
        with t[2]:
            s["cover_letter"] = st.text_area("Cover letter (edit freely)", s.get("cover_letter", ""), height=320)
        with t[3]:
            s["outreach"] = st.text_area("Recruiter / referral message", s.get("outreach", ""), height=160)
        with t[4]:
            for q in s.get("questions", []):
                st.markdown(f"**[{q.get('type', '')}] {q['question']}**  \n_Hint: {q.get('hint', '')}_")
        with t[5]:
            for st_ in result.steps:
                st.markdown(f"{st_.n}. `{st_.tool}` by **{st_.chosen_by}** — {st_.thought}  \n→ {st_.observation}")

        md = (f"# {req.get('role', '')} at {req.get('company', '')}\n\nMatch: {m.get('score', 0)}%\n\n## Resume bullets\n"
              + "\n".join(f"- {b['tailored']}" for b in s.get("bullets", []))
              + f"\n\n## Cover letter\n{s.get('cover_letter', '')}\n\n## Outreach\n{s.get('outreach', '')}\n\n## Interview prep\n"
              + "\n".join(f"- {q['question']} (hint: {q.get('hint', '')})" for q in s.get("questions", [])))
        a, b_ = st.columns(2)
        a.download_button("Download as Markdown", md, file_name="application.md")
        if b_.button("Approve and save to tracker"):
            app_id = db.save(st.session_state.con, s, st.session_state.get("jd_url", ""))
            st.success(f"Saved as application #{app_id}. Nothing was sent anywhere.")

with tab_tracker:
    rows = db.list_all(st.session_state.con)
    if not rows:
        st.info("No saved applications yet. Approve one on the first tab.")
    else:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        app_id = c1.selectbox("Application", [r["id"] for r in rows])
        status = c2.selectbox("New status", db.STATUSES)
        if c3.button("Update"):
            db.update_status(st.session_state.con, app_id, status)
            st.rerun()
        st.caption("Note: on free Streamlit hosting this database resets when the app restarts. Download important drafts.")
