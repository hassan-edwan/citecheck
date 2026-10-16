"""Review the real-citation labels in a spreadsheet.   Owner: Modeling.

    python -m tools.realcite_sheet export                      # -> data/realcite/raw/review.csv (git-ignored)
    # open it in Excel / Sheets: read the citing sentence and the numbered abstract, fill your_label (blank = agree)
    python -m tools.realcite_sheet import --reviewer Ritika     # -> updates data/realcite/labels.jsonl

The sheet holds the citing sentences, so it stays in data/realcite/raw/ and is never committed.  On import, a blank
your_label accepts the draft label; a different label replaces it and keeps the draft in `draft_label`.  Either way
the row gets reviewed_by = the reviewer.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from eval.run_realcite import GOLD
from src.data_io import find_scifact_dir, load_corpus, load_jsonl

COLUMNS = ["pair_id", "cited_title", "citing_sentence", "cited_abstract_numbered", "draft_label", "draft_claims",
           "your_label", "note"]


def export(labels_path, raw_path, data_dir, out_path):
    raw = {r["pair_id"]: r for r in load_jsonl(raw_path)}
    by_id = {d.doc_id: d for d in load_corpus(find_scifact_dir(data_dir) / "corpus.jsonl")}
    labels = load_jsonl(labels_path) if Path(labels_path).exists() else []
    labelled = {lab["pair_id"]: lab for lab in labels}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel opens UTF-8 correctly
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        for pid, r in raw.items():
            lab = labelled.get(pid, {})
            doc = by_id[r["cited_doc_id"]]
            w.writerow({"pair_id": pid, "cited_title": doc.title, "citing_sentence": r["context"],
                        "cited_abstract_numbered": " ".join(f"[{i}] {s}" for i, s in enumerate(doc.sentences)),
                        "draft_label": lab.get("label", ""), "draft_claims": " | ".join(lab.get("attributed_claims", [])),
                        "your_label": "", "note": lab.get("note", "")})
    return len(raw)


def import_review(labels_path, raw_path, sheet_path, reviewer):
    raw = {r["pair_id"]: r for r in load_jsonl(raw_path)}
    labels = {lab["pair_id"]: lab for lab in load_jsonl(labels_path)} if Path(labels_path).exists() else {}
    changed = accepted = 0
    with open(sheet_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pid, mine = row["pair_id"], (row.get("your_label") or "").strip().upper()
            lab = labels.get(pid)
            if lab is None and not mine:
                continue  # neither drafted nor labelled now
            if mine and mine not in GOLD:
                raise ValueError(f"{pid}: your_label {mine!r} is not one of {' | '.join(GOLD)}")
            if lab is None:
                r = raw[pid]
                lab = {"pair_id": pid, "cited_doc_id": r["cited_doc_id"], "citing_paper_id": r["citing_paper_id"],
                       "context_sha1": r["context_sha1"], "label": mine, "attributed_claims": [], "evidence_sentences": [],
                       "annotator": reviewer}
            elif mine and mine != lab["label"]:
                lab = {**lab, "draft_label": lab["label"], "label": mine}
                changed += 1
            else:
                accepted += 1
            if (row.get("note") or "").strip():
                lab["note"] = row["note"].strip()
            labels[pid] = {**lab, "reviewed_by": reviewer}
    Path(labels_path).write_text("".join(json.dumps(v) + "\n" for v in labels.values()), encoding="utf-8")
    return accepted, changed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["export", "import"])
    ap.add_argument("--labels", default="data/realcite/labels.jsonl")
    ap.add_argument("--raw", default="data/realcite/raw/candidates.jsonl")
    ap.add_argument("--sheet", default="data/realcite/raw/review.csv")
    ap.add_argument("--data-dir", default="data/scifact")
    ap.add_argument("--reviewer", default=None)
    a = ap.parse_args(argv)
    if a.action == "export":
        print(f"[saved] {a.sheet}: {export(a.labels, a.raw, a.data_dir, Path(a.sheet))} pairs")
    else:
        if not a.reviewer:
            raise SystemExit("--reviewer NAME is required: every reviewed label records who checked it.")
        acc, chg = import_review(a.labels, a.raw, a.sheet, a.reviewer)
        print(f"[saved] {a.labels}: {acc} draft labels accepted, {chg} changed, reviewer {a.reviewer}")


if __name__ == "__main__":
    main()
