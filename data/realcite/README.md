# Data card: real-citation set (`data/realcite/`)

Owner: Modeling. Purpose: measure the citation audit on **real citing sentences**, which SciFact's rewritten, atomic
claims do not represent. This is the "small labelled set of REAL citations" planned for Session 9 in
`docs/roadmap_to_demo_day.md`, started early so the audit can be measured at all.

## What a pair is

A sentence from a published paper that cites a paper in our SciFact corpus, plus that cited paper's abstract.

| part | where it comes from | in git? |
|---|---|---|
| citing sentence, citing paper id/title | Semantic Scholar Graph API, `citations` with `contexts` (`tools/collect_citations.py`) | **no**: `data/realcite/raw/` is git-ignored; re-collect with the same `--seed` |
| cited abstract | our local SciFact corpus (ODC-By 1.0) | no (re-download with `data.fetch_scifact`) |
| labels | annotators, `labels.jsonl` | yes |

Cited papers that any SciFact TRAIN claim uses are excluded, so neither the fine-tuned verifier nor the trained
sentence selector has seen them. Each cited paper's Semantic Scholar title is checked against the corpus title.
`context_sha1` in each label pins the exact sentence that was labelled; `eval/run_realcite.py` refuses a pair whose
re-collected sentence differs.

## Labels (`labels.jsonl`, one JSON object per line)

```
{"pair_id": "...", "cited_doc_id": "...", "citing_paper_id": "...", "context_sha1": "...",
 "label": "SUPPORT | PARTIAL | CONTRADICT | NEI | NOT_A_CLAIM",
 "attributed_claims": ["what the sentence attributes to the cited paper", ...],
 "evidence_sentences": [indices into the cited abstract],
 "annotator": "...", "reviewed_by": null | "name", "note": "..."}
```

How to label (judge the cited **abstract** only, since that is all the system reads):

- **SUPPORT**: the abstract establishes everything the sentence attributes to the cited paper, at the same strength.
- **PARTIAL**: it establishes some attributed claims, or a weaker version (association where the sentence says cause,
  mice where it says humans), and nothing in it contradicts the rest.
- **CONTRADICT**: the abstract reports the opposite of an attributed claim.
- **NEI**: the abstract is about the topic but does not establish the claim (it may be in the full text).
- **NOT_A_CLAIM**: the paper is cited for a method, dataset, tool or definition, or only the citing authors' own
  results are stated; there is nothing to verify.

Hedges count: "may reduce" is supported by a reported reduction; "reduces" is only PARTIAL if the abstract says "may".

## Who labelled it

Labels marked `"annotator": "ai-draft"` are **draft labels written by an AI assistant**. They are there to make
human labelling faster, never as ground truth: an AI-labelled set cannot fairly judge an AI system. Every label needs
`reviewed_by` set by a teammate who read the sentence and the abstract and agreed or corrected it.
`eval/run_realcite.py` flags any result computed while labels are unreviewed. Target: about 50 reviewed pairs, with
two teammates on a shared subset of at least 15 to report agreement (Cohen's kappa).

## Current set (collected 2026-10-16, `--n-papers 40 --per-paper 2 --seed 0`)

80 pairs from 40 cited papers. Draft labels (`ai-draft`, none reviewed yet): SUPPORT 28, NOT_A_CLAIM 21, NEI 18,
PARTIAL 12, CONTRADICT 1.

What the draft labelling found about real citation contexts:

- **About a quarter are not usable claims** (21 of 80): 9 cite a method, tool or material ("as described previously",
  an atlas, plasmids), and 12 are not real citing sentences at all: reference-list entries, author lists,
  correspondence addresses, table fragments, or sentences Semantic Scholar attached to the wrong reference.
  An audit that judges every context as a claim will produce confident verdicts on noise.
- **PARTIAL is common** (12 of 80): citing sentences often add something the abstract does not establish (a population,
  a mechanism, "first", "routinely recommended"), or rest on a cluster of citations of which this paper covers part.
- **NEI often means "the full text may say it"**: review abstracts (IKK complex, LTP history) rarely list the specific
  facts papers cite them for.
- Three labels flag a possible contradiction for the reviewer to decide (pairs on bladder-cancer progression, amyloid-
  first ordering, and PRT6 and leaf senescence); only the PRT6 one is drafted as CONTRADICT.

## How the labels are used in the system

The set is split by cited paper into a **tune** half (20 papers, 40 pairs) and a **test** half (20 papers, 40 pairs);
`eval/run_realcite.realcite_split` fixes the split. Only the tune half is used to build anything:

1. **Citation-context gate** (`src/citation_gate.py`), the audit's first stage: rules that skip reference entries,
   addresses and headers, author lists, table fragments, cut-off clauses, method citations and sentences without a
   citation marker. Rules were written from tune-half examples. Held-out test half: caught **6 of 10** non-claims and
   wrongly skipped **0 of 30** real claims (tune half: 8 of 11, 0 of 29). Caveat: these labels were written by reading
   all 80 pairs, and the test-half totals were seen once during development (not which pairs), so the test half guards
   against tuned rules, not against the labeller's familiarity. It cannot catch contexts Semantic Scholar attached to
   the wrong reference.
2. **Audit abstain threshold** (`audit_tau`): `run_realcite --tune-tau` picks it on the tune half (best macro-F1 with
   false-SUPPORT at most `--fs-cap`, default 0.10) and reports the test half. `--promote-audit` writes it to
   `deployed_config.json`; with unreviewed labels it refuses unless `--allow-unreviewed`, which is recorded as
   `audit_labels_reviewed: false`.

Both depend on unreviewed AI drafts. When the labels are reviewed, re-run `run_realcite --tune-tau` and re-check the
gate numbers before quoting them.
