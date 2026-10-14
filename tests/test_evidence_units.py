"""Evidence units: what the verifier reads per paper (window / abstract / sentence)."""
import json

import pytest

from eval import finetune_nli, run_eval
from src.config import load_config
from src.data_io import load_claims, load_corpus
from src.evidence import LexicalSentenceScorer, aggregate_sentence_probs, score_claim_docs
from src.pipeline import CiteCheckPipeline, build_pipeline
from src.retrieval import BM25Retriever
from src.schema import Doc, Hit, Probs, SUPPORT
from tests.test_pipeline import Scripted
from tests.test_run_eval import base_args

DOC = Doc(doc_id="d1", title="T", sentences=(
    "Background on bones.",
    "Zorvex increased bone density in mice.",
    "Methods were standard.",
    "Zorvex also raised density in rats.",
))
CLAIM = "Zorvex increases bone density."


def test_aggregate_takes_most_decisive_sentence_and_its_allies():
    probs = [Probs(0.1, 0.8, 0.1), Probs(0.9, 0.05, 0.05), Probs(0.2, 0.2, 0.6), Probs(0.6, 0.3, 0.1)]
    doc, idxs = aggregate_sentence_probs(probs)
    assert doc == probs[1] and idxs == [1, 3]
    doc, idxs = aggregate_sentence_probs([Probs(0.1, 0.8, 0.1), Probs(0.2, 0.7, 0.1)])
    assert idxs == [1]  # nothing reaches 0.5: show the strongest sentence anyway
    assert aggregate_sentence_probs([]) == (Probs(0.0, 1.0, 0.0), [])


def test_units_feed_the_verifier_different_premises():
    seen = []

    class Recording(Scripted):
        def predict_pairs(self, pairs):
            seen.extend(p for p, _ in pairs)
            return super().predict_pairs(pairs)

    v = Recording({"increased bone": Probs(0.9, 0.05, 0.05), "raised density": Probs(0.7, 0.2, 0.1)})
    items = [(CLAIM, [Hit(DOC, 1.0, 1)])]

    seen.clear()
    ab = score_claim_docs(items, v, unit="abstract")[0][0]
    assert seen == [" ".join(DOC.sentences)] and ab.sentence_idxs == [0, 1, 2, 3]

    seen.clear()
    se = score_claim_docs(items, v, unit="sentence")[0][0]
    assert seen == list(DOC.sentences)                      # one verifier call per sentence
    assert se.probs.support == 0.9 and se.sentence_idxs == [1, 3]
    assert se.premise == DOC.sentences[1] + " " + DOC.sentences[3]

    seen.clear()
    wi = score_claim_docs(items, v, m=2, unit="window")[0][0]
    assert len(seen) == 1 and len(wi.sentence_idxs) == 2

    # the oracle-rationale override restricts every unit to the gold sentences
    seen.clear()
    score_claim_docs(items, v, unit="sentence", gold_sentences=[{"d1": [2]}])
    assert seen == [DOC.sentences[2]]
    with pytest.raises(ValueError):
        score_claim_docs(items, v, unit="paragraph")


def test_sentence_unit_handles_a_paper_without_abstract():
    empty = Doc(doc_id="e", title="Only a title", sentences=())
    out = score_claim_docs([(CLAIM, [Hit(empty, 1.0, 1), Hit(DOC, 0.5, 2)])], Scripted({}), unit="sentence")[0]
    assert out[0].premise == "Only a title" and out[0].sentence_idxs == [] and len(out) == 2


def test_pipeline_uses_configured_unit(fixture_dir, monkeypatch, tmp_path):
    corpus = load_corpus(fixture_dir / "corpus.jsonl")
    pipe = CiteCheckPipeline(Scripted({"zorvex increased": Probs(0.9, 0.05, 0.05)}), BM25Retriever(corpus),
                             LexicalSentenceScorer.from_docs(corpus), tau=0.5, evidence_unit="sentence")
    v = pipe.verify_claim(CLAIM, docs=[DOC])
    assert v.label == SUPPORT and v.evidence[0].sentences == [DOC.sentences[1]]

    dep = tmp_path / "deployed_config.json"
    dep.write_text(json.dumps({"run_name": "x", "verifier": "lexical", "evidence_unit": "abstract"}))
    assert load_config(dep).evidence_unit == "abstract"
    monkeypatch.setenv("CITECHECK_EVIDENCE_UNIT", "sentence")
    cfg = load_config(dep)
    assert cfg.evidence_unit == "sentence" and not cfg.validated
    monkeypatch.setenv("CITECHECK_VERIFIER", "lexical")
    monkeypatch.setenv("CITECHECK_DATA_DIR", str(fixture_dir))
    assert build_pipeline().evidence_unit == "sentence"


@pytest.mark.parametrize("unit", ["abstract", "sentence"])
def test_pair_builders_match_the_unit(fixture_dir, unit):
    corpus = load_corpus(fixture_dir / "corpus.jsonl")
    by_id = {d.doc_id: d for d in corpus}
    claims = load_claims(fixture_dir / "claims_train.jsonl")
    pairs = finetune_nli.PAIR_BUILDERS[unit](claims, by_id, BM25Retriever(corpus), LexicalSentenceScorer.from_docs(corpus))
    assert {p[2] for p in pairs} == {"support", "contradict", "neutral"}
    abstracts = {" ".join(d.sentences) for d in corpus}
    sentences = {s for d in corpus for s in d.sentences}
    if unit == "abstract":  # every gold paper appears whole, with its label
        assert any(p[0] in abstracts and p[2] != "neutral" for p in pairs)
    else:  # neutral premises are single sentences
        assert all(p[0] in sentences for p in pairs if p[2] == "neutral")


@pytest.mark.parametrize("unit", ["abstract", "sentence"])
def test_train_and_evaluate_each_unit(fixture_dir, tmp_path, unit):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests.tiny_model import make_tiny_nli

    tiny = make_tiny_nli(tmp_path / "base")
    out = tmp_path / "ft"
    finetune_nli.main(["--base", tiny, "--data-dir", str(fixture_dir), "--out", str(out), "--epochs", "1", "--max-steps", "2",
                       "--batch-size", "4", "--device", "cpu", "--tune-n", "3", "--unit", unit])
    assert json.loads((out / "train_log.json").read_text())["args"]["unit"] == unit
    r = run_eval.main(base_args(fixture_dir, tmp_path / "res", unit, verifier="nli",
                                extra=["--model", str(out), "--device", "cpu", "--tune-n", "3", "--evidence-unit", unit]))
    assert r["config"]["evidence_unit"] == unit and r["config"]["trained_on_unit"] == unit
    assert f"[{unit}]" in (tmp_path / "res" / "LATEST.md").read_text()
