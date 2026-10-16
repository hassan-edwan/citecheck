"""CiteCheck web app.   Owner: Engineering (Sakshaat).

    streamlit run app/streamlit_app.py

Environment (all optional):
  CITECHECK_VERIFIER / _RETRIEVER / _TAU / _K / _DATA_DIR   override the evaluated configuration (=> UNVALIDATED banner)
  CITECHECK_LOG=logs/usage.jsonl                            where usage events go
  CITECHECK_LOG_ENABLED=0                                   turn logging off
  CITECHECK_DEMO_EXAMPLES=demo/demo_examples.json           live-demo examples (written by `python -m eval.pick_demo_examples`)
  CITECHECK_MAILTO=you@umd.edu                              polite-pool e-mail for OpenAlex requests
  S2_API_KEY=...                                            optional Semantic Scholar key (fewer rate limits)
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.ingest import Fetcher, IngestError, paper_from_text  # noqa: E402
from src.pipeline import build_pipeline  # noqa: E402
from src.relevance import RelevanceScorer  # noqa: E402
from src.schema import CONTRADICT, DISPLAY, NEI, SUPPORT  # noqa: E402
from src.text_utils import truncate  # noqa: E402
from src.usage_log import UsageLogger  # noqa: E402

st.set_page_config(page_title="CiteCheck", page_icon="🔎", layout="wide")

EXAMPLE = {
    "claim": "Zorvex supplementation increases bone density in mice.",
    "title": "Zorvex supplementation and skeletal density in mice (INVENTED EXAMPLE)",
    "abstract": ("Bone loss is a major concern in ageing populations. We fed adult mice a diet supplemented with zorvex for "
                 "twelve weeks. Zorvex supplementation significantly increased femoral bone density compared with controls. "
                 "The effect was dose dependent and was not observed in mice fed a standard diet."),
}


def load_demo_examples(path=None) -> list:
    """Seeded, pre-checked SciFact dev examples for the live demo (main + backups + the honest failure, if picked)."""
    path = Path(path or os.environ.get("CITECHECK_DEMO_EXAMPLES", "demo/demo_examples.json"))
    try:
        j = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = list(j.get("examples") or []) + ([j["failure"]] if j.get("failure") else [])
    return [e for e in items if e.get("claim") and e.get("abstract")]


@st.cache_resource(show_spinner="Loading models (first start can take a minute)…")
def get_pipeline():
    return build_pipeline()


@st.cache_resource
def get_fetcher():
    return Fetcher()


def logger() -> UsageLogger:
    if "logger" not in st.session_state:
        st.session_state.logger = UsageLogger(
            path=os.environ.get("CITECHECK_LOG", "logs/usage.jsonl"),
            enabled=os.environ.get("CITECHECK_LOG_ENABLED", "1") != "0",
            participant=st.query_params.get("p"),
        )
        st.session_state.logger.log("session_start")
    return st.session_state.logger


pipe, fetcher, log = get_pipeline(), get_fetcher(), logger()


# ----------------------------------------------------------------------------------------------
# Sidebar: honesty about what is running + researcher panel
# ----------------------------------------------------------------------------------------------
def sidebar():
    m = pipe.meta
    st.sidebar.header("Model status")
    if m.get("demo_only"):
        st.sidebar.error("DEMO BACKEND (word overlap). Verdicts are NOT reliable. The neural model did not load.")
    elif m.get("validated"):
        lo, hi = m.get("dev_macro_f1_ci95") or (None, None)
        st.sidebar.success(
            f"Validated configuration `{m.get('deployed_run')}`\n\n"
            f"SciFact dev macro-F1 **{m['dev_macro_f1']:.2f}** (95% CI {lo:.2f} to {hi:.2f}); "
            f"false-SUPPORT rate **{m['dev_false_support_rate']:.2f}**."
        )
    else:
        st.sidebar.warning("UNVALIDATED configuration: no evaluated run is deployed, or settings were overridden. "
                           "Do not quote numbers for this setup.")
    if m.get("validated") and not m.get("demo_only"):
        st.sidebar.caption("This is the exact model whose accuracy we report (eval/results). The numbers are from SciFact, "
                           "a biomedical dataset.")
    for w in pipe.warnings:
        st.sidebar.info(w)
    st.sidebar.caption(f"verifier `{m.get('verifier')}` · retriever `{m.get('retriever')}` · tau {m.get('tau')}")

    with st.sidebar.expander("Researcher panel (user tests)"):
        code = st.text_input("Participant code (P01, P02…; never a name)", value=log.participant or "", key="participant")
        consent = st.checkbox("Participant consented to logging what they type", key="consent")
        log.participant, log.log_text = (code.strip() or None), bool(consent)
        task = st.selectbox("Task", ["T1", "T2", "T3"], key="task_id")
        c1, c2, c3 = st.columns(3)
        if c1.button("▶ Start", key="task_start"):
            st.session_state.task_t0 = time.time()
            log.log("task_start", task_id=task)
        for col, key, label, outcome in ((c2, "task_ok", "✔ Done", "completed"), (c3, "task_fail", "✖ Gave up", "gave_up")):
            if col.button(label, key=key):
                t0 = st.session_state.get("task_t0")
                log.log("task_end", task_id=task, outcome=outcome, elapsed_s=round(time.time() - t0, 1) if t0 else None)
                st.session_state.task_t0 = None
        st.caption(f"logging to `{log.path}` · text logging {'ON' if log.log_text else 'off'}")


sidebar()

st.title("🔎 CiteCheck")
st.markdown("**Does the paper you cite really support your claim?** CiteCheck reads the paper's abstract and shows you "
            "the sentences that decide it, so you can check the verdict yourself.")
_h1, _h2, _h3 = st.columns(3)
_h1.markdown("**① Your claim**  \nThe sentence you want to cite the paper for.")
_h2.markdown("**② The paper**  \nA DOI, arXiv id or link, or paste its abstract.")
_h3.markdown("**③ The verdict**  \nSupports, contradicts, or not enough evidence, with the sentences that decide it.")
tab_check, tab_rel, tab_audit, tab_about = st.tabs(
    ["Check a citation", "Is this paper relevant?", "Audit a paper (experimental)", "About & limits"])


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
HEADLINE = {
    SUPPORT: ("✅", "The abstract supports your claim"),
    CONTRADICT: ("❌", "The abstract contradicts your claim"),
    NEI: ("❔", "Not enough evidence in the abstract to decide"),
}
MEANING = {
    SUPPORT: "The model judged that the sentences below back up your claim. That is a strong hint, not proof: "
             "read them before you cite.",
    CONTRADICT: "The sentences below appear to conflict with your claim. Check whether the paper's population and "
                "conditions match yours before you drop or change the citation.",
    NEI: "Nothing in the abstract clearly supports or contradicts the claim. That does NOT mean the claim is false: "
         "CiteCheck only reads abstracts, and it says so instead of guessing when it is unsure.",
}


def render_verdict(v, key_prefix="v"):
    box = {SUPPORT: st.success, CONTRADICT: st.error, NEI: st.warning}[v.label]
    icon, headline = HEADLINE[v.label]
    box(f"#### {icon} {headline}\n**Verdict: {v.display_label}**  ·  model confidence {v.confidence:.2f}")
    st.progress(min(max(float(v.confidence), 0.0), 1.0),
                text=f"Model confidence {v.confidence:.0%}: a score from the model, not a guarantee")
    st.caption(MEANING[v.label] + (f"  ({v.note})" if v.label == NEI and v.note else ""))
    if v.evidence:
        st.markdown("**Evidence the model read**")
    for i, e in enumerate(v.evidence, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. [{e.title}]({e.url})**" if e.url else f"**{i}. {e.title}**")
            for s in e.sentences:
                st.markdown(f"> {s}")
            st.caption(f"How the model read these sentences: supports {e.probs.support:.0%} · "
                       f"neutral {e.probs.neutral:.0%} · contradicts {e.probs.contradict:.0%}")
    with st.expander("Technical details"):
        st.json({"latency_ms": round(v.latency_ms), "backend": v.backend, "abstained": v.abstained})


def source_inputs(prefix, allow_corpus):
    options = ["The paper I'm citing (DOI / arXiv / link)", "Paste an abstract"] + (["Search the SciFact corpus"] if allow_corpus else [])
    choice = st.radio("Where should I look for evidence?", options, horizontal=True, key=f"{prefix}_src",
                      help="Give the paper you cite. If it can't be fetched (no internet, no open abstract), paste its abstract.")
    ident = title = abstract = ""
    if choice == options[0]:
        ident = st.text_input("DOI, arXiv id or link", key=f"{prefix}_ident", placeholder="10.1038/nature14539  or  arXiv:2004.14974",
                              help="We look the paper up in OpenAlex, Semantic Scholar and arXiv and read its abstract.")
    elif choice == options[1]:
        title = st.text_input("Title (optional)", key=f"{prefix}_title")
        abstract = st.text_area("Abstract", key=f"{prefix}_abstract", height=160,
                                placeholder="Paste the paper's abstract here. Only the abstract is read.")
    return options.index(choice), ident, title, abstract


def resolve_paper(mode, ident, title, abstract):
    if mode == 0:
        return fetcher.fetch_paper(ident)
    return paper_from_text(title, abstract)


# ----------------------------------------------------------------------------------------------
# Tab 1: check a citation
# ----------------------------------------------------------------------------------------------
def fill_example(claim, title, abstract, gold=None):
    """Widget values must be set BEFORE the widgets are drawn, so this runs above them."""
    st.session_state.claim = claim
    st.session_state.chk_src = "Paste an abstract"
    st.session_state.chk_title = title
    st.session_state.chk_abstract = abstract
    st.session_state.demo_gold = (claim, gold) if gold else None
    st.session_state.pop("verdict", None)


def clear_check():
    """'Start over' (runs as a callback, before the widgets are drawn)."""
    for k in ("claim", "chk_title", "chk_abstract", "chk_ident", "fb_comment"):
        st.session_state[k] = ""
    for k in ("verdict", "demo_gold", "feedback_sent", "check_problem"):
        st.session_state.pop(k, None)


def missing_input(claim, mode, ident, abstract):
    """A friendly reason the check cannot run yet, or None."""
    if not claim.strip():
        return "Enter a claim to check: one sentence, as you would write it in your paper."
    if mode == 0 and not ident.strip():
        return "Enter the DOI, arXiv id or link of the paper you cite, or choose “Paste an abstract”."
    if mode == 1 and not abstract.strip():
        return "Paste the paper's abstract, or choose “The paper I'm citing” to look it up by DOI / arXiv id."
    return None


with tab_check:
    left, right = st.columns([1, 1], gap="large")
    with left:
        demo = load_demo_examples()
        if demo:
            with st.expander("Demo examples (SciFact dev, picked by seed; for presenters, not for participants)"):
                names = [f"{e.get('slot', '?')} · SciFact claim #{e.get('claim_id', '?')} · {e.get('role', '')}" for e in demo]
                pick = st.selectbox("Example", range(len(demo)), format_func=lambda i: names[i], key="demo_pick")
                if st.button("Load this example", key="btn_demo"):
                    e = demo[pick]
                    fill_example(e["claim"], e.get("title", ""), e["abstract"], e.get("gold"))
                    log.log("demo_example", slot=e.get("slot"), claim_id=e.get("claim_id"))
        if st.button("Fill in an invented example", key="btn_example", help="Loads a made-up claim and abstract to try the tool."):
            fill_example(EXAMPLE["claim"], EXAMPLE["title"], EXAMPLE["abstract"])
        claim = st.text_area("Your claim: the sentence you want the paper to support", key="claim", height=90,
                             placeholder="e.g. Antiretroviral therapy reduces the incidence of tuberculosis.",
                             help="One short, self-contained statement works best. Leave out the citation marker.")
        if len(claim) > 300:
            st.caption("Tip: CiteCheck works best on one short claim. Consider splitting long sentences.")
        mode, ident, title, abstract = source_inputs("chk", pipe.has_corpus)

        b1, b2 = st.columns([3, 1])
        clicked = b1.button("Check citation", type="primary", key="btn_check", use_container_width=True)
        b2.button("Start over", key="btn_clear", on_click=clear_check, use_container_width=True)
        if clicked:
            t0 = time.time()
            problem = missing_input(claim, mode, ident, abstract)
            if problem:
                st.session_state.pop("verdict", None)
                st.warning(problem)
                log.log("error", where="verify", message=problem[:200])
            else:
                try:
                    with st.spinner("Reading the abstract and weighing the evidence…"):
                        if mode == 2:
                            verdict = pipe.verify_claim(claim)
                        else:
                            verdict = pipe.verify_against_paper(claim, resolve_paper(mode, ident, title, abstract))
                    st.session_state.verdict = verdict
                    st.session_state.pop("feedback_sent", None)
                    log.log("verify", claim=claim, mode=["paper", "pasted", "corpus"][mode], verdict=verdict.label,
                            confidence=round(verdict.confidence, 3), abstained=verdict.abstained, n_evidence=len(verdict.evidence),
                            latency_ms=round(verdict.latency_ms), wall_ms=round(1000 * (time.time() - t0)),
                            verifier=verdict.backend.get("verifier"), validated=verdict.backend.get("validated"))
                except (IngestError, ValueError) as e:
                    st.session_state.pop("verdict", None)
                    st.error(f"Couldn't check this citation: {e}")
                    log.log("error", where="verify", message=str(e)[:200])

    with right:
        v = st.session_state.get("verdict")
        if v is None:
            st.info("**Your verdict will appear here.**  \nEnter a claim and the paper on the left, then press "
                    "**Check citation**. New here? Press **Fill in an invented example** first.")
        else:
            render_verdict(v)
            claim_gold = st.session_state.get("demo_gold")
            if claim_gold and claim_gold[0].strip() == v.claim:          # only if the example's claim was not edited
                gold = claim_gold[1]
                mark = "matches" if gold == v.label else "does NOT match"
                st.info(f"SciFact annotators' label for this claim and paper: **{DISPLAY.get(gold, gold)}**; the model {mark} it.")
            st.markdown("**Do you agree with this verdict?**")
            c1, c2, _ = st.columns([1, 1, 2])
            for col, key, label, val in ((c1, "fb_yes", "👍 Agree", "agree"), (c2, "fb_no", "👎 Disagree", "disagree")):
                if col.button(label, key=key):
                    st.session_state.feedback_sent = val
                    log.log("feedback", agrees=val == "agree", verdict=v.label)
            if st.session_state.get("feedback_sent"):
                st.success(f"Thanks: recorded “{st.session_state.feedback_sent}”.")
                comment = st.text_input("Anything we got wrong or confusing? (optional)", key="fb_comment")
                if comment and st.button("Send comment", key="fb_send"):
                    log.log("comment", comment=comment)
                    st.info("Comment saved.")

# ----------------------------------------------------------------------------------------------
# Tab 2: relevance
# ----------------------------------------------------------------------------------------------
with tab_rel:
    st.markdown("Describe what you are working on, then point me at a paper. This is a quick skim aid, **secondary** to citation checking.")
    direction = st.text_area("Your research direction", key="direction", height=80,
                             placeholder="e.g. Using NLP to verify scientific claims against cited evidence")
    rmode, rident, rtitle, rabstract = source_inputs("rel", False)
    if st.button("How relevant is it?", key="btn_rel"):
        if not direction.strip():
            st.warning("Describe your research direction first (one or two sentences).")
            st.stop()
        try:
            paper = resolve_paper(rmode, rident, rtitle, rabstract)
            r = RelevanceScorer().score(direction, paper)
            st.info(f"**{r.band}**  (score {r.score:.2f}, method: {r.method})")
            st.write("Matched terms: " + (", ".join(r.matched_terms) or "none"))
            st.caption(f"{paper.title} · {r.note}")
            log.log("relevance", direction=direction, band=r.band, score=round(r.score, 3))
        except (IngestError, ValueError) as e:
            st.error(str(e))

# ----------------------------------------------------------------------------------------------
# Tab 3: audit
# ----------------------------------------------------------------------------------------------
with tab_audit:
    st.warning("**Experimental.** We have no labelled data for this setting. Citing sentences in real papers are messier than "
               "the atomic claims we validated on, and many papers expose no citing sentences at all. Treat rows as leads to check.")
    aud_id = st.text_input("DOI, arXiv id or link of the paper to audit", key="aud_ident")
    n_refs = st.slider("References to check", 3, 15, 8, key="aud_n")
    if st.button("Audit its citations", key="btn_audit"):
        if not aud_id.strip():
            st.warning("Enter the DOI, arXiv id or link of the paper whose references you want to check.")
            st.stop()
        try:
            bar = st.progress(0.0, text="Fetching the reference list…")
            refs = fetcher.fetch_references(aud_id, limit=max(n_refs, 10))
            rows = pipe.audit(refs, max_items=n_refs, progress=lambda i, n: bar.progress(i / n, text=f"Checking reference {i} of {n}…"))
            bar.empty()
            table = [{
                "Cited paper": truncate(r.reference.cited.title, 80),
                "Citing sentence": truncate(r.reference.context, 140),
                "Claims checked": sum(c.verdict is not None for c in r.claims),
                "Verdict": r.verdict.display_label if r.verdict else f"skipped ({r.skipped_reason})",
                "Confidence": round(r.verdict.confidence, 2) if r.verdict else None,
            } for r in rows]
            st.dataframe(table)
            for r in rows:
                if len(r.claims) > 1 or any(c.status != "checked" for c in r.claims):
                    with st.expander(f"Claims for: {truncate(r.reference.cited.title, 70)}"):
                        for c in r.claims:
                            status = c.verdict.display_label if c.verdict else c.status.replace("_", " ")
                            st.markdown(f"- **{status}**: {c.claim or c.context}" + (f"  \n  _{c.note}_" if c.note else ""))
            log.log("audit", identifier=aud_id, n_rows=len(rows), n_checked=sum(1 for r in rows if r.verdict))
        except (IngestError, ValueError) as e:
            st.error(str(e))

# ----------------------------------------------------------------------------------------------
# Tab 4: about
# ----------------------------------------------------------------------------------------------
with tab_about:
    st.markdown("""
**What this is.** A course project (UMD DATA/MSML 641). It judges whether *evidence sentences from a paper's abstract*
entail, contradict, or are neutral to a claim, using a pretrained natural-language-inference model and a confidence threshold.

**What a wrong answer costs.** The worst error is a false **SUPPORTS**: it can make you trust a citation that does not hold up.
That is why the system abstains (“not enough evidence”) unless it is confident, and why every verdict shows the sentences it used.

**Known limits.**
- Trained/evaluated on **SciFact**, which is **biomedical**. Do not assume it works for other fields.
- It reads **abstracts only**; support that lives in a paper's results or methods is invisible to it.
- Claims work best when they are short, self-contained statements.
- Relevance scores are uncalibrated hints.
- It is a second pair of eyes. **You** are responsible for what you cite.

**Data & privacy.** Evaluation data: SciFact (claims CC BY 4.0, abstracts ODC-By 1.0). Usage logs store no personal
information; the text you type is only logged in research mode with your consent, with e-mails/phone numbers redacted.
""")
