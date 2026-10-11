# CiteCheck

A paper-judging tool for researchers. Submit a paper (link or DOI) along with a short note on your research direction, and get back a verdict on two things:

1. **Relevance** — does this paper actually fit your research direction?
2. **Citation support** — do the papers it cites genuinely support the claims they're attached to?

PDF upload is a possible future input method, still deciding on OCR handling.

## Who this is for

Students and researchers who read and cite papers weekly. Today they either skim abstracts and guess, or read full papers to check relevance, and when writing their own work, they cite sources without an easy way to verify the cited paper actually supports the claim it's attached to.

## Why this over the alternatives

- Manually reading full papers doesn't scale past a handful a week
- Google Scholar / Semantic Scholar surface papers but don't judge fit to a specific research direction or check citation support
- Asking a chatbot directly isn't grounded in retrieval and can hallucinate support that isn't there

CiteCheck is narrower and grounded: retrieval-backed verdicts, not open-ended summaries.

**Value proposition:** We help researchers judge whether a paper is worth reading and whether its citations hold up, faster than reading it themselves or trusting an ungrounded chatbot summary.

**The MVP job** (see `docs/mvp_scope.md`): *check whether the paper I cite really supports my claim, and show me the sentences that decide it.* Relevance and citation audit are secondary/experimental.

## North-star metric

F1 on citation-support judgments (SciFact / SciFact-Open) vs. a zero-shot baseline. Retrieval quality tracked separately via precision@k. Precisely: claim-level macro-F1 on SciFact dev with a 95% bootstrap CI, plus the guardrail *false-SUPPORT rate* (see `docs/north_star.md`). Every reported number comes from a committed run in `eval/results/`.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows PowerShell; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                                      # all tests should pass (a few skip if torch is missing)

python -m data.fetch_scifact                          # download SciFact (claims CC BY 4.0, abstracts ODC-By 1.0)
python -m eval.run_eval --name zero_shot_nli --retriever bm25 --verifier nli --oracle --promote
streamlit run app/streamlit_app.py                    # the sidebar must say "Validated configuration"
```

No internet, or no model download? `CITECHECK_VERIFIER=lexical streamlit run app/streamlit_app.py` runs a clearly-labelled DEMO backend.
Command line: `python -m src.main verify --claim "..." --abstract "..."` (see `python -m src.main --help`).

## Live demo (Oct 6)

```bash
python -m eval.pick_demo_examples --include-failure   # once (Data & Eval): seeded SciFact dev examples -> demo/
python -m tools.demo_check --online                   # first run on the presenting laptop (caches the model)
python -m tools.demo_check                            # Wi-Fi-off rehearsal: must end "overall: PASS"
streamlit run app/streamlit_app.py                    # Check a citation -> "Demo examples" -> Load -> Check citation
```

The presenter card with what should appear is `demo/DEMO_SCRIPT.md`; the check result is `demo/DEMO_CHECK.md`.

## Team

| Hat | Owner | Accountable for |
|---|---|---|
| Product | Vyom | The user, the roadmap, the lean canvas, the pitch |
| Engineering | Sakshaat | Architecture, code review, repo health, deployment |
| Data and Evaluation | Sahil | Data and licensing, the evaluation harness, metrics, error analysis |
| Users and Research | Ritika | Recruiting users, running sessions, capturing raw evidence |
| Modeling | Hassan | NLP models, fine-tuning, embeddings, prompt engineering, and inference optimization |

## Repo structure

- `src/` — core pipeline: `retrieval.py` (BM25 / dense / hybrid / rerank), `evidence.py` (sentence selection), `nli.py` (verifiers), `verdict.py` (abstain rule), `pipeline.py`, `ingest.py` (DOI/arXiv), `main.py` (CLI)
- `app/` — Streamlit app
- `data/` — SciFact download + data card (`data/README.md`: licenses, limits, biases)
- `eval/` — evaluation harness, error analysis, plots, fine-tuning (`eval/README.md`)
- `evidence/` — raw user-test evidence per session (`evidence/README.md`)
- `reports/` — weekly progress reports (`sessionNN.md`)
- `docs/` — MVP scope, user stories, north-star definition, midterm plan, roadmap
- `tools/` — `preflight.py` (report check), user-test tooling
- `tests/` — pytest suite (uses a small SYNTHETIC fixture; its numbers are never reported)

## Working agreement

One issue and one branch per task; a teammate approves every PR; no direct pushes to `main`. Before a report deadline run
`python -m tools.preflight --session NN --gh --repo Lord2709/citecheck`.

## Status

Early stage. Problem statement and team roles are set. The end-to-end pipeline, app and evaluation harness are in place; see `eval/results/LATEST.md` for what has actually been measured and `reports/` for the current weekly state. Not yet deployed for strangers (planned for Session 7).
