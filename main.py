from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
import html
import json
import os
import re
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ResumeIQ - Resume Screening", page_icon="📄", layout="wide")

# ----------------------------------------------------------------------------
# Backend configuration (identical to the notebook)  -- UNCHANGED
# ----------------------------------------------------------------------------
DB_DIR = os.getenv("RESUMEIQ_DB_DIR", "ResumeRAG_1000")   # Chroma persist_directory
LLM_MODEL = "gpt-4.1"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 1000
RETRIEVE_K = 30          # retriever_similarity: search_kwargs={"k": 30}
SHORTLIST_N = 4          # shortlisted_candidates = ranked_candidates[:4]


def _api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            key = None
    return key


@st.cache_resource(show_spinner="Connecting to resume vector store...")
def load_backend():
    from langchain_chroma import Chroma
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    key = _api_key()
    model = ChatOpenAI(model=LLM_MODEL, api_key=key)
    embedding = OpenAIEmbeddings(model=EMBED_MODEL, api_key=key, dimensions=EMBED_DIMS)
    vectore_store = Chroma(embedding_function=embedding, persist_directory=DB_DIR)
    retriever = vectore_store.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVE_K})
    metas = vectore_store.get(include=["metadatas"])["metadatas"] or []
    sources = sorted({m["source"] for m in metas if m and m.get("source")})
    return model, retriever, sources


def evaluate_candidate(model, job_description, candidate_name, candidate_resume):
    """Same prompt and JSON contract as the notebook's evaluate_candidate()."""
    prompt = f"""
You are an expert AI recruitment evaluator.

Evaluate the candidate ONLY against the provided Job Description.

Do not assume or invent any information that is not explicitly present
in the candidate's resume.

JOB DESCRIPTION:
{job_description}

CANDIDATE NAME:
{candidate_name}

CANDIDATE RESUME:
{candidate_resume}

Evaluate the candidate based on:
- Required skills
- Technical skills
- Tools and technologies
- Relevant experience
- Projects
- Education, when relevant

Return ONLY valid JSON in exactly this format:

{{
    "candidate": "{candidate_name}",
    "match_score": 0,
    "matching_skills": [],
    "missing_skills": [],
    "experience_fit": "",
    "reason": ""
}}

Scoring:
90-100 = Excellent match
80-89 = Strong match
70-79 = Good match
60-69 = Partial match
Below 60 = Weak match

The match_score must be between 0 and 100.
"""
    result_text = model.invoke(prompt).content
    result_text = re.sub(r"```json|```", "", result_text).strip()
    return json.loads(result_text)


def screen(model, retriever, user_input, progress):
    """retrieve -> group by resume -> evaluate -> rank -> shortlist."""
    retrieved_docs = retriever.invoke(
        f"""
    Identify candidates relevant to this Job Description:

    {user_input}
    """
    )
    candidate_resumes = {}
    for doc in retrieved_docs:
        candidate_resumes.setdefault(doc.metadata.get("source"), []).append(doc.page_content)

    candidate_results, failed = [], []
    for i, (source, chunks) in enumerate(candidate_resumes.items(), 1):
        candidate_name = re.split(r"[\\/]", source)[-1]
        progress(i - 1, len(candidate_resumes), candidate_name)
        try:
            result = evaluate_candidate(model, user_input, candidate_name, "\n\n".join(chunks))
            result["source"] = source
            candidate_results.append(result)
        except Exception as exc:  # bad JSON / API error for one resume should not kill the run
            failed.append(f"{candidate_name}: {exc}")
    progress(len(candidate_resumes), len(candidate_resumes), "done")

    ranked_candidates = sorted(candidate_results, key=lambda x: x["match_score"], reverse=True)
    shortlisted_candidates = ranked_candidates[:SHORTLIST_N]
    return candidate_results, ranked_candidates, shortlisted_candidates, failed


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
esc = html.escape
INK = "#111827"      # dark text
NAVY = "#0b2e59"     # headings
BLUE = "#1f5fa8"     # buttons / bars


def clean_name(filename):
    n = re.sub(r"\.pdf$", "", filename, flags=re.I).replace("_", " ")
    n = re.sub(r"(?i)resume|\bcv\b", "", n)
    n = re.sub(r"\s+", " ", n).strip(" -")
    return n or filename


def role_from_jd(jd, title):
    if title.strip():
        return title.strip()
    for line in jd.splitlines():
        line = line.strip(" \t•-*")
        if line:
            return (line[:60] + "...") if len(line) > 60 else line
    return "Untitled role"


def fmt(x):
    return f"{x:g}"


def section(title):
    st.markdown(f'<div class="sec">{esc(title)}</div>', unsafe_allow_html=True)


def tags(items, cls=""):
    if not items:
        return '<span class="muted">None reported</span>'
    return "".join(f'<span class="tag {cls}">{esc(str(s))}</span>' for s in items)


def bar_chart(data, y, x, x_title, color=BLUE, label_fmt="d", pct=False):
    """Horizontal bar chart on a WHITE background with dark, readable text."""
    d = data.copy()
    d["label"] = d[x].map(lambda v: f"{fmt(v)}%" if pct else f"{v:.0f}")
    top = 100 if pct else max(d[x].max(), 1)
    base = alt.Chart(d).encode(
        y=alt.Y(f"{y}:N", sort=alt.EncodingSortField(x, order="descending"), title=None,
                axis=alt.Axis(labelLimit=320, labelFontSize=13, labelColor=INK, labelPadding=8,
                              ticks=False, domain=False)),
        x=alt.X(f"{x}:Q", title=x_title,
                scale=alt.Scale(domain=[0, top * 1.12], nice=False),
                axis=alt.Axis(values=list(range(0, int(top) + 1, 20)) if pct else alt.Undefined,
                              format=label_fmt, labelColor=INK, titleColor=INK, labelFontSize=12,
                              titleFontSize=12, gridColor="#e5e7eb", domainColor="#9ca3af",
                              tickMinStep=1)),
        tooltip=[y, x],
    )
    bars = base.mark_bar(color=color, size=24)
    text = base.mark_text(align="left", dx=6, fontSize=13, fontWeight="bold", color=INK).encode(text="label:N")
    return ((bars + text)
            .properties(height=max(90, 44 * len(d)), background="#ffffff")
            .configure_view(fill="#ffffff", strokeWidth=0))


# ----------------------------------------------------------------------------
# UI styling (light theme forced so text stays readable in dark mode too)
# ----------------------------------------------------------------------------
st.markdown(
    f"""
<style>
.stApp {{background:#ffffff;color:{INK};}}
header[data-testid="stHeader"], footer, #MainMenu {{display:none;}}
.block-container {{padding:1.2rem 2rem 2rem 2rem; max-width:1200px;}}
.stApp p, .stApp li, .stApp label, .stApp span {{color:{INK};}}
.stApp h1, .stApp h2, .stApp h3 {{color:{NAVY};}}
.rq-title {{border-bottom:2px solid {NAVY};padding-bottom:6px;margin-bottom:6px;}}
.stApp .rq-title h1 {{font-size:28px;font-weight:700;margin:0;padding:0;color:{NAVY};}}
.stApp .rq-title p {{font-size:14px;margin:2px 0 0 0;color:#374151;}}
.sec {{font-size:17px;font-weight:600;color:{NAVY};border-bottom:1px solid #d1d5db;padding-bottom:4px;margin:26px 0 12px 0;}}
.kpi {{border:1px solid #d1d5db;padding:10px 14px;background:#f9fafb;}}
.stApp .kpi span {{font-size:12px;color:#4b5563;display:block;}}
.kpi div {{font-size:24px;font-weight:600;color:{NAVY};margin-top:2px;}}
.stApp .tag {{display:inline-block;font-size:13px;background:#eff6ff;border:1px solid #bfdbfe;color:#1e3a8a;padding:2px 8px;margin:3px 5px 3px 0;}}
.stApp .tag.x {{background:#fef2f2;border-color:#fecaca;color:#7f1d1d;}}
.muted {{color:#6b7280;font-size:13px;}}
.stApp .lbl {{font-size:13px;font-weight:600;color:#374151;margin:12px 0 2px 0;}}
.txt {{font-size:14px;line-height:1.55;color:{INK};white-space:pre-wrap;}}
table.rq {{width:100%;border-collapse:collapse;font-size:13.5px;}}
table.rq th {{background:#f3f4f6;color:{INK};text-align:left;padding:7px 10px;border-bottom:1px solid #d1d5db;}}
table.rq td {{color:{INK};padding:7px 10px;border-bottom:1px solid #e5e7eb;}}
.stApp input, .stApp textarea {{background:#ffffff !important;color:{INK} !important;border-radius:2px !important;}}
.stApp div[data-baseweb="select"] > div {{background:#ffffff;color:{INK};border-radius:2px;}}
.stApp div[data-baseweb="select"] * {{color:{INK};}}
.stApp .stButton > button {{background:{BLUE};color:#ffffff;border:0;border-radius:2px;padding:.4rem 1.3rem;font-size:14px;}}
.stApp .stButton > button p {{color:#ffffff;}}
.stApp .stButton > button:hover {{background:#174a85;color:#ffffff;}}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# 1. Header
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="rq-title"><h1>ResumeIQ</h1><p>AI Resume Screening &amp; Candidate Ranking</p></div>',
    unsafe_allow_html=True,
)

if not _api_key():
    st.error("OPENAI_API_KEY is not set. Set it as an environment variable (or in .streamlit/secrets.toml) and restart.")
    st.stop()
if not os.path.isdir(DB_DIR):
    st.error(f"Chroma store '{DB_DIR}' not found. Run from the folder that contains it, or set RESUMEIQ_DB_DIR.")
    st.stop()

model, retriever, all_sources = load_backend()

ss = st.session_state
ss.setdefault("runs", [])
if "_goto" in ss:                       # jump to the newest run after screening
    ss["run_idx"] = ss.pop("_goto")

# ----------------------------------------------------------------------------
# 2. Job description / screening
# ----------------------------------------------------------------------------
section("Job Description / Screening")
title_in = st.text_input("Job Role", key="jd_title", placeholder="e.g. AI / ML Automation Engineer")
jd_in = st.text_area("Job Description", height=180, key="jd_text",
                     placeholder="Paste the recruiter's job description here.")
if st.button("Screen Candidates"):
    if not jd_in.strip():
        st.warning("Enter a Job Description first.")
    else:
        bar = st.progress(0.0, text="Retrieving relevant resumes...")
        try:
            def progress(i, n, name):
                bar.progress(i / max(n, 1), text=f"Evaluating {clean_name(name)} ({i}/{n})")
            cr, rc, sc, failed = screen(model, retriever, jd_in, progress)
        except Exception as exc:
            bar.empty()
            st.error(f"Screening failed: {exc}")
        else:
            bar.empty()
            ss["runs"].append(dict(role=role_from_jd(jd_in, title_in), jd=jd_in,
                                   time=datetime.now().strftime("%d %b %Y %H:%M"),
                                   candidate_results=cr, ranked=rc, shortlisted=sc))
            ss["_goto"] = len(ss["runs"]) - 1
            ss["_failed"] = failed
            st.rerun()
for f in ss.pop("_failed", []):
    st.warning(f"Skipped - {f}")

# Pick which screening run to display (only shown when there is more than one)
run = None
if ss["runs"]:
    if len(ss["runs"]) > 1:
        idx = st.selectbox("Showing results for", range(len(ss["runs"])),
                           format_func=lambda i: f"{ss['runs'][i]['time']} - {ss['runs'][i]['role']}",
                           key="run_idx")
    else:
        idx = 0
    run = ss["runs"][idx]

ranked = run["ranked"] if run else []
shortlisted = run["shortlisted"] if run else []
scores = [float(c["match_score"]) for c in ranked]

# ----------------------------------------------------------------------------
# 3. Summary
# ----------------------------------------------------------------------------
section("Summary")
kpis = [
    ("Total Resumes", f"{len(all_sources)}"),
    ("Candidates Evaluated", f"{len(ranked)}" if run else "-"),
    ("Shortlisted Candidates", f"{len(shortlisted)}" if run else "-"),
    ("Average Match Score", f"{sum(scores) / len(scores):.0f}%" if scores else "-"),
    ("Best Match Score", f"{fmt(max(scores))}%" if scores else "-"),
]
for col, (label, val) in zip(st.columns(5), kpis):
    col.markdown(f'<div class="kpi"><span>{label}</span><div>{val}</div></div>', unsafe_allow_html=True)

if not run:
    st.markdown('<p class="muted" style="margin-top:14px">Enter a job description and click '
                '<b>Screen Candidates</b> to see the ranking.</p>', unsafe_allow_html=True)
    st.stop()

# ----------------------------------------------------------------------------
# 4. Candidate ranking
# ----------------------------------------------------------------------------
section("Candidate Ranking")
rank_df = pd.DataFrame({
    "Candidate": [clean_name(c["candidate"]) for c in ranked],
    "Match Score": scores,
})
st.altair_chart(bar_chart(rank_df, "Candidate", "Match Score", "Match Score (%)", pct=True),
                use_container_width=True, theme=None)
short_src = {c["source"] for c in shortlisted}
st.markdown(f'<p class="muted">Shortlisted: top {SHORTLIST_N} by match score '
            f'({len(shortlisted)} candidate{"s" if len(shortlisted) != 1 else ""}).</p>',
            unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 5. Candidate details
# ----------------------------------------------------------------------------
section("Candidate Details")
pick = st.selectbox("Select candidate", range(len(ranked)),
                    format_func=lambda p: f"{p + 1}. {clean_name(ranked[p]['candidate'])}", key="cand_pick")
cand = ranked[pick]
status = "Shortlisted" if cand["source"] in short_src else "Not shortlisted"
st.markdown(
    f'<div class="lbl">Match Score</div><div class="txt"><b>{fmt(float(cand["match_score"]))}%</b>'
    f' &nbsp;<span class="muted">({status})</span></div>'
    f'<div class="lbl">Matching Skills</div><div>{tags(cand["matching_skills"])}</div>'
    f'<div class="lbl">Missing Skills</div><div>{tags(cand["missing_skills"], "x")}</div>'
    f'<div class="lbl">Experience Fit</div><div class="txt">{esc(cand["experience_fit"])}</div>'
    f'<div class="lbl">Reason for Match</div><div class="txt">{esc(cand["reason"])}</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# 6. Skill gap
# ----------------------------------------------------------------------------
section("Skill Gap")
left, right = st.columns(2)
for col, title, key, colr in [(left, "Top Matching Skills", "matching_skills", BLUE),
                              (right, "Top Missing Skills", "missing_skills", "#b45309")]:
    with col:
        st.markdown(f'<div class="lbl" style="margin-top:0">{title}</div>', unsafe_allow_html=True)
        counts = pd.Series([s for c in ranked for s in c[key]]).value_counts().head(8)
        if counts.empty:
            st.markdown('<span class="muted">No data.</span>', unsafe_allow_html=True)
        else:
            cd = counts.rename_axis("Skill").reset_index(name="Candidates")
            st.altair_chart(bar_chart(cd, "Skill", "Candidates", "Number of candidates", color=colr),
                            use_container_width=True, theme=None)

# ----------------------------------------------------------------------------
# 7. Screening history
# ----------------------------------------------------------------------------
section("Screening History")
rows = "".join(
    f"<tr><td>{esc(r['time'])}</td><td>{esc(r['role'])}</td><td>{len(r['ranked'])}</td>"
    f"<td>{fmt(max((float(c['match_score']) for c in r['ranked']), default=0))}%</td></tr>"
    for r in reversed(ss["runs"])
)
st.markdown("<table class='rq'><tr><th>Date</th><th>Job Role</th><th>Candidates Evaluated</th>"
            f"<th>Best Match Score</th></tr>{rows}</table>", unsafe_allow_html=True)
with st.expander("View job description for the selected run"):
    st.text(run["jd"])