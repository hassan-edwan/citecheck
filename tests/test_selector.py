"""Evidence-sentence selector: cross-encoder scorer, training, and the selection metric in run_eval."""
import json

import pytest

from eval import run_eval
from src.data_io import load_claims, load_corpus
from src.evidence import LexicalSentenceScorer, build_sentence_scorer, selection_recall
from tests.test_run_eval import base_args


def test_selection_recall_needs_a_complete_rationale():
    sents = ["alpha beta", "gamma delta", "zorvex bone density", "zorvex mice"]
    lex = LexicalSentenceScorer()
    assert selection_recall("zorvex bone density", sents, [[2]], 1, lex) is True
    assert selection_recall("zorvex bone density", sents, [[0, 2]], 1, lex) is False   # half a rationale is not enough
    assert selection_recall("zorvex bone density", sents, [], 1, lex) is None
    with pytest.raises(ValueError):
        build_sentence_scorer("bm25")


def test_selector_pairs_label_rationale_sentences(fixture_dir):
    from eval.train_selector import build_selector_pairs
    from src.retrieval import BM25Retriever

    corpus = load_corpus(fixture_dir / "corpus.jsonl")
    by_id = {d.doc_id: d for d in corpus}
    claims = load_claims(fixture_dir / "claims_train.jsonl")
    pairs = build_selector_pairs(claims, by_id, BM25Retriever(corpus), LexicalSentenceScorer.from_docs(corpus))
    pos = {(c, s) for c, s, y in pairs if y == 1}
    want = {(c.claim, by_id[d].sentences[i]) for c in claims for d in c.gold_docs for i in c.rationale_sentence_idxs(d)}
    assert pos == want and any(y == 0 for *_, y in pairs)


def test_train_selector_and_use_it_in_run_eval(fixture_dir, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from eval import train_selector
    from tests.tiny_model import make_tiny_nli

    base = make_tiny_nli(tmp_path / "base")
    out = tmp_path / "sel"
    train_selector.main(["--base", base, "--data-dir", str(fixture_dir), "--out", str(out), "--epochs", "1",
                         "--max-steps", "2", "--batch-size", "8", "--device", "cpu", "--tune-n", "3", "--max-length", "64"])
    log = json.loads((out / "train_log.json").read_text())
    assert log["kind"] == "sentence_selector" and len(log["heldout_claim_ids"]) == 3 and log["val_log"]

    scorer = build_sentence_scorer(f"cross:{out}", device="cpu")
    scores = scorer.score("Longer sleep improves recall.", ["Sleep helps memory.", "Unrelated sentence.", "x"])
    assert len(scores) == 3 and all(isinstance(s, float) for s in scores)

    r = run_eval.main(base_args(fixture_dir, tmp_path / "res", "sel", extra=["--sentence-scorer", f"cross:{out}", "--device", "cpu"]))
    assert r["config"]["sentence_scorer"] == f"cross:{out}"
    ss = r["sentence_selection"]
    assert ss["n_gold_papers"] > 0 and 0 <= ss["rationale_recall_at_m"] <= 1
    lex = run_eval.main(base_args(fixture_dir, tmp_path / "res", "lex"))
    assert lex["config"]["sentence_scorer"] == "lexical" and "Sentence selection" in (tmp_path / "res" / "lex" / "summary.md").read_text()


def test_pipeline_falls_back_to_word_overlap_when_selector_is_missing(monkeypatch, fixture_dir, tmp_path):
    from src.pipeline import build_pipeline

    monkeypatch.setenv("CITECHECK_VERIFIER", "lexical")
    monkeypatch.setenv("CITECHECK_DATA_DIR", str(fixture_dir))
    monkeypatch.setenv("CITECHECK_SENTENCE_SCORER", f"cross:{tmp_path / 'missing'}")
    pipe = build_pipeline()
    assert isinstance(pipe.scorer, LexicalSentenceScorer) and any("Sentence selector" in w for w in pipe.warnings)
    assert not pipe.meta["validated"]
