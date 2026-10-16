"""Real-citation set: collection (fake Semantic Scholar) and evaluation of the audit on labelled pairs."""
import json
import random

import pytest

from eval import run_realcite
from src.data_io import load_claims, load_corpus
from tools import collect_citations as C


class FakeS2:
    """Answers like the Graph API; titles match for every doc except 'bad'."""

    def __init__(self, by_id, bad=()):
        self.by_id, self.bad, self.calls = by_id, set(bad), []

    def get(self, url, params=None):
        self.calls.append(url)
        doc_id = url.split("CorpusId:")[1].split("/")[0]
        if url.endswith("/citations"):
            return {"data": [
                {"citingPaper": {"paperId": "cp1", "title": "Citer one", "year": 2021}, "intents": ["result"],
                 "contexts": [f"Earlier work showed {self.by_id[doc_id].title.lower()} matters a great deal [4]."]},
                {"citingPaper": {"paperId": "cp2", "title": "Citer two", "year": 2022}, "intents": [],
                 "contexts": ["short", f"We used the method of [9] described in {doc_id} for our analysis here."]},
            ]}
        return {"title": "something else entirely" if doc_id in self.bad else self.by_id[doc_id].title}


def test_eligible_docs_exclude_everything_train_uses(fixture_dir):
    corpus = load_corpus(fixture_dir / "corpus.jsonl")
    train = load_claims(fixture_dir / "claims_train.jsonl")
    used = {d for c in train for d in list(c.gold_docs) + list(c.cited_doc_ids)}
    eligible = C.eligible_docs(corpus, train)
    assert eligible and not (set(eligible) & used)


def test_collect_checks_titles_and_samples(fixture_dir):
    corpus = load_corpus(fixture_dir / "corpus.jsonl")
    by_id = {d.doc_id: d for d in corpus}
    ids = sorted(by_id)[:3]
    rows, checked = C.collect(by_id, ids, FakeS2(by_id, bad=[ids[0]]), n_papers=2, per_paper=2, rng=random.Random(0),
                              log=lambda *_: None)
    assert checked == 3 and {r["cited_doc_id"] for r in rows} == set(ids[1:])   # title mismatch skipped
    assert all(len(r["context"]) >= 40 and r["context_sha1"] == C.context_sha1(r["context"]) for r in rows)
    assert len({r["pair_id"] for r in rows}) == len(rows)
    assert C.title_match("Zorvex and bones", "zorvex AND bones!") == 1.0


def write_set(tmp_path, fixture_dir, labels):
    corpus = load_corpus(fixture_dir / "corpus.jsonl")
    by_id = {d.doc_id: d for d in corpus}
    rows, _ = C.collect(by_id, sorted(by_id)[:2], FakeS2(by_id), 2, 2, random.Random(0), log=lambda *_: None)
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(r) + "\n" for r in rows))
    lab = tmp_path / "labels.jsonl"
    lab.write_text("".join(json.dumps({"pair_id": r["pair_id"], "cited_doc_id": r["cited_doc_id"], "context_sha1": r["context_sha1"],
                                       "label": labels(r["context"]) if callable(labels) else labels[i % len(labels)],
                                       "annotator": "test", "reviewed_by": None}) + "\n"
                           for i, r in enumerate(rows)))
    return raw, lab, rows


def test_run_realcite_scores_whole_sentence_and_extracted(fixture_dir, tmp_path, monkeypatch):
    raw, lab, rows = write_set(tmp_path, fixture_dir, lambda ctx: "NOT_A_CLAIM" if "method of" in ctx else "SUPPORT")

    class FakeLLM:
        def __init__(self, model_name, **_):
            self.model_name = model_name

        def generate(self, prompts):  # method citations: nothing attributed; others: the whole sentence's first words
            out = []
            for p in prompts:
                sent = p.split("Citing sentence: ")[-1].split("\nAnswer:")[0]
                out.append('{"claims": []}' if "method of" in sent else json.dumps(
                    {"claims": [{"claim": sent, "source_quote": sent, "role": "result", "needs_more_context": False}]}))
            return out

    monkeypatch.setattr("src.reasoning.LocalLLM", FakeLLM)
    monkeypatch.setenv("CITECHECK_VERIFIER", "lexical")
    r = run_realcite.main(["--name", "rc", "--labels", str(lab), "--raw", str(raw), "--data-dir", str(fixture_dir),
                           "--out", str(tmp_path / "res"), "--claim-extractor", "fake", "--n-boot", "50"])
    assert r["n_pairs"] == len(rows) and r["labels_unreviewed"] == len(rows)
    assert set(r["systems"]) == {"whole_sentence", "extracted"} and "extracted_vs_whole_sentence" in r
    assert r["systems"]["extracted"]["method_citations_skipped"] == 1.0   # every method citation was left unjudged
    preds = (tmp_path / "res" / "rc" / "predictions.jsonl").read_text()
    assert all(row["context"] not in preds for row in rows)                # citing text is never written out


def test_run_realcite_refuses_changed_sentences_and_bad_labels(fixture_dir, tmp_path):
    raw, lab, rows = write_set(tmp_path, fixture_dir, ["SUPPORT"])
    first = json.loads(lab.read_text().splitlines()[0])
    lab.write_text(json.dumps({**first, "context_sha1": "0" * 40}) + "\n")
    with pytest.raises(ValueError, match="differs"):
        run_realcite.load_pairs(lab, raw)
    lab.write_text(json.dumps({**first, "label": "MAYBE"}) + "\n")
    with pytest.raises(ValueError, match="unknown label"):
        run_realcite.load_pairs(lab, raw)


def test_review_sheet_round_trip(fixture_dir, tmp_path):
    import csv

    from tools import realcite_sheet as S

    raw, lab, rows = write_set(tmp_path, fixture_dir, ["SUPPORT"])
    sheet = tmp_path / "review.csv"
    assert S.export(lab, raw, fixture_dir, sheet) == len(rows)
    table = list(csv.DictReader(open(sheet, encoding="utf-8-sig")))
    assert table[0]["cited_abstract_numbered"].startswith("[0] ") and table[0]["draft_label"] == "SUPPORT"
    table[0]["your_label"] = "partial"          # disagree with the draft on one row, accept the rest
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, S.COLUMNS)
        w.writeheader()
        w.writerows(table)
    assert S.import_review(lab, raw, sheet, "Ritika") == (len(rows) - 1, 1)
    out = {json.loads(l)["pair_id"]: json.loads(l) for l in lab.read_text().splitlines()}
    first = out[table[0]["pair_id"]]
    assert first["label"] == "PARTIAL" and first["draft_label"] == "SUPPORT" and first["reviewed_by"] == "Ritika"
    assert all(v["reviewed_by"] == "Ritika" for v in out.values())
    table[0]["your_label"] = "maybe"
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, S.COLUMNS)
        w.writeheader()
        w.writerows(table)
    with pytest.raises(ValueError):
        S.import_review(lab, raw, sheet, "Ritika")
