"""End-to-end tests of the harness on the SYNTHETIC fixture (plumbing only; numbers are meaningless)."""
import json
from pathlib import Path

import pytest

from eval import error_analysis, plots, run_eval
from src.data_io import load_claims


def base_args(fixture_dir, out, name, verifier="lexical", extra=()):
    return ["--name", name, "--data-dir", str(fixture_dir), "--out", str(out), "--retriever", "bm25",
            "--verifier", verifier, "--no-cache", "--n-boot", "50", *extra]


def test_full_run_and_files(fixture_dir, tmp_path):
    res = run_eval.main(base_args(fixture_dir, tmp_path, "lexical", extra=["--oracle"]))
    d = tmp_path / "lexical"
    assert {p.name for p in d.iterdir()} == {"results.json", "predictions.jsonl", "summary.md"}
    assert res["synthetic_fixture"] is True and res["partial"] is False
    assert res["claim_level"]["n"] == 10 and res["claim_level"]["n_mixed_excluded"] == 0
    assert res["tau_tuning"]["split"] == "train" and res["tau_tuning"]["leak"] is False
    assert 0 <= res["claim_level"]["macro_f1"] <= 1
    assert res["retrieval"]["recall@3"] >= 0.8                       # BM25 finds the gold paper on the fixture
    att = res["error_attribution"]
    assert set(att) >= {"end_to_end", "oracle_doc", "oracle_rationale", "loss_from_retrieval", "loss_from_sentence_selection"}
    assert att["oracle_doc"] >= att["end_to_end"] - 1e-9             # perfect retrieval can't hurt when BM25 already hits gold
    assert abs(att["loss_from_retrieval"] - (att["oracle_doc"] - att["end_to_end"])) < 1e-12
    assert "SYNTHETIC" in (d / "summary.md").read_text()
    assert "SYNTHETIC" in (tmp_path / "LATEST.md").read_text()
    preds = [json.loads(l) for l in (d / "predictions.jsonl").read_text().splitlines()]
    assert len(preds) == 10 and {"id", "gold", "pred", "confidence", "docs"} <= set(preds[0])


def test_deterministic(fixture_dir, tmp_path):
    a = run_eval.main(base_args(fixture_dir, tmp_path / "a", "x"))
    b = run_eval.main(base_args(fixture_dir, tmp_path / "b", "x"))
    assert a["claim_level"]["macro_f1"] == b["claim_level"]["macro_f1"]
    assert a["config"]["tau"] == b["config"]["tau"]


def test_majority_floor_and_paired_comparison(fixture_dir, tmp_path):
    maj = run_eval.main(base_args(fixture_dir, tmp_path, "majority", verifier="majority"))
    assert maj["claim_level"]["coverage"] == 0.0 and maj["claim_level"]["false_support_rate"] == 0.0
    lex = run_eval.main(base_args(fixture_dir, tmp_path, "lexical", extra=["--compare-with", str(tmp_path / "majority")]))
    (cmp_,) = lex["paired_comparisons"]
    assert cmp_["vs"] == "majority" and cmp_["n"] == 10 and set(cmp_) >= {"diff", "ci95", "p_a_better"}
    assert lex["claim_level"]["macro_f1"] > maj["claim_level"]["macro_f1"]


def test_refuses_leakage_test_split_and_promotion(fixture_dir, tmp_path):
    with pytest.raises(SystemExit, match="Refusing to tune"):
        run_eval.main(base_args(fixture_dir, tmp_path, "leak", extra=["--tune-split", "dev"]))
    ok = run_eval.main(base_args(fixture_dir, tmp_path, "leak_ok", extra=["--tune-split", "dev", "--allow-leak"]))
    assert ok["tau_tuning"]["leak"] is True                                     # recorded, not hidden
    with pytest.raises(SystemExit, match="promoted"):
        run_eval.main(base_args(fixture_dir, tmp_path, "p", extra=["--promote"]))   # synthetic runs can't be deployed
    assert not (tmp_path / "deployed_config.json").exists()


def test_limit_flags_partial(fixture_dir, tmp_path):
    r = run_eval.main(base_args(fixture_dir, tmp_path, "small", extra=["--limit", "4"]))
    assert r["partial"] is True and r["claim_level"]["n"] == 4


def test_error_analysis_and_plots(fixture_dir, tmp_path):
    run_eval.main(base_args(fixture_dir, tmp_path, "lexical"))
    error_analysis.main(["--run", str(tmp_path / "lexical"), "--top", "3"])
    md = (tmp_path / "lexical" / "errors.md").read_text()
    assert "Error analysis" in md and "your category" in md
    assert (tmp_path / "lexical" / "errors.csv").exists()
    plots.main(["--results", str(tmp_path), "--run", "lexical", "--include-synthetic"])
    figs = {p.name for p in (tmp_path / "figures").iterdir()}
    assert {"f1_by_run.png", "retrieval_recall.png", "confusion_lexical.png", "tau_lexical.png"} <= figs
    plots.main(["--results", str(tmp_path), "--out", str(tmp_path / "f2")])      # synthetic skipped by default
    assert not (tmp_path / "f2" / "f1_by_run.png").exists()


def test_bucket_logic():
    b = error_analysis.bucket
    assert b({"gold": "NEI", "pred": "SUPPORT", "docs": []}) == "FALSE_SUPPORT"
    assert b({"gold": "CONTRADICT", "pred": "SUPPORT", "docs": []}) == "WRONG_DIRECTION"
    assert b({"gold": "NEI", "pred": "CONTRADICT", "docs": []}) == "FALSE_CONTRADICT"
    assert b({"gold": "SUPPORT", "pred": "NEI", "docs": [{"doc_id": "7"}], "gold_docs": ["1"]}) == "RETRIEVAL_MISS"
    assert b({"gold": "SUPPORT", "pred": "NEI", "docs": [{"doc_id": "1"}], "gold_docs": ["1"]}) == "MISSED_BY_VERIFIER"
    assert b({"gold": "NEI", "pred": "NEI", "docs": []}) is None


def test_real_transformer_end_to_end_and_finetune(fixture_dir, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from eval import finetune_nli
    from src.data_io import load_corpus
    from src.evidence import LexicalSentenceScorer
    from src.retrieval import BM25Retriever
    from tests.tiny_model import make_tiny_nli

    corpus = load_corpus(fixture_dir / "corpus.jsonl")
    pairs = finetune_nli.build_training_pairs(load_claims(fixture_dir / "claims_train.jsonl"), {d.doc_id: d for d in corpus},
                                              BM25Retriever(corpus), LexicalSentenceScorer.from_docs(corpus))
    kinds = {p[2] for p in pairs}
    assert kinds == {"support", "contradict", "neutral"}
    assert all(p[0] and p[1] for p in pairs)
    # the gold-rationale positive for claim 101 uses exactly the annotated sentence
    assert ("Volunteers who slept eight hours recalled more word pairs than those kept awake.",
            "Longer sleep improves recall of word pairs.", "support") in pairs

    tiny = make_tiny_nli(tmp_path / "base")
    out = tmp_path / "ft"
    finetune_nli.main(["--base", tiny, "--data-dir", str(fixture_dir), "--out", str(out), "--epochs", "2",
                       "--max-steps", "3", "--batch-size", "4", "--device", "cpu", "--max-length", "64",
                       "--tune-n", "3", "--no-match-inference"])
    assert (out / "train_log.json").exists() and (out / "config.json").exists()
    log = json.loads((out / "train_log.json").read_text())
    heldout = {c.id for c in run_eval.tau_tuning_claims(load_claims(fixture_dir / "claims_train.jsonl"), 3, 0)}
    assert "claims_dev" not in json.dumps(log["args"]) and set(log["heldout_claim_ids"]) == heldout
    trained = [c for c in load_claims(fixture_dir / "claims_train.jsonl") if c.id not in heldout]
    assert log["n_pairs"] == len(finetune_nli.build_training_pairs(
        trained, {d.doc_id: d for d in corpus}, BM25Retriever(corpus), LexicalSentenceScorer.from_docs(corpus)))
    assert log["val_log"] and log["val_log"][0]["val_pair_macro_f1"] is not None
    # the fine-tuned checkpoint plugs straight into the evaluation harness, tuning tau on the held-out claims only
    r = run_eval.main(base_args(fixture_dir, tmp_path / "res", "ft", verifier="nli",
                                extra=["--model", str(out), "--device", "cpu", "--tune-n", "3"]))
    assert r["config"]["verifier"].startswith("nli:") and r["timing"]["n_params"] > 0
    assert r["tau_tuning"]["leak"] is False and r["tau_tuning"]["n_tuning_claims_seen_in_finetuning"] == 0
    assert json.loads((tmp_path / "res" / "ft" / "train_log.json").read_text()) == log  # travels with the results
    # tuning tau on claims the checkpoint was trained on is refused, or recorded as a leak with --allow-leak
    with pytest.raises(SystemExit):
        run_eval.main(base_args(fixture_dir, tmp_path / "res", "ft2", verifier="nli", extra=["--model", str(out), "--device", "cpu"]))
    r = run_eval.main(base_args(fixture_dir, tmp_path / "res", "ft3", verifier="nli",
                                extra=["--model", str(out), "--device", "cpu", "--allow-leak"]))
    assert r["tau_tuning"]["leak"] is True and r["tau_tuning"]["n_tuning_claims_seen_in_finetuning"] > 0


def test_match_inference_pairs_use_selector_output(fixture_dir):
    from eval import finetune_nli
    from src.data_io import load_corpus
    from src.evidence import LexicalSentenceScorer

    corpus = load_corpus(fixture_dir / "corpus.jsonl")
    by_id, scorer = {d.doc_id: d for d in corpus}, LexicalSentenceScorer.from_docs(corpus)
    claims = load_claims(fixture_dir / "claims_train.jsonl")
    base = finetune_nli.build_training_pairs(claims, by_id, None, scorer, m=2)
    more = finetune_nli.build_training_pairs(claims, by_id, None, scorer, m=2, match_inference=True, gold_doc_neutral=1)
    assert set(base) <= set(more) and len(more) > len(base)
    for premise, claim, cls in set(more) - set(base):
        assert cls in finetune_nli.CLASSES and premise and claim


def test_tau_tuning_claims_matches_session5_sample(fixture_dir):
    """The refactored helper must pick the same claims the inline Session 5 code did (results stay comparable)."""
    import numpy as np

    claims = sorted(load_claims(fixture_dir / "claims_train.jsonl"), key=lambda c: c.id)
    pick = sorted(np.random.default_rng(0).choice(len(claims), 3, replace=False))
    assert [c.id for c in run_eval.tau_tuning_claims(claims, 3, 0)] == [claims[i].id for i in pick]
    assert len(run_eval.tau_tuning_claims(claims, 100, 0)) == len(claims)


def test_versions_come_from_package_metadata():
    """Session 5 bug: rank_bm25 has no __version__, so provenance recorded null."""
    from importlib import metadata

    from eval.run_eval import _versions

    v = _versions()
    assert v["rank_bm25"] == metadata.version("rank-bm25")
    assert v["numpy"] == metadata.version("numpy")


def test_error_report_names_the_driver_paper():
    """errors.md used to print the rank-1 title next to sentences taken from the driver paper (#100-style errors)."""
    from types import SimpleNamespace

    from eval.error_analysis import _driver_label

    corpus = {"1": SimpleNamespace(title="Gold paper"), "2": SimpleNamespace(title="Off-topic paper")}
    rec = {"driver_doc": "2", "gold_docs": ["1"]}
    label = _driver_label(rec, {"doc_id": "2", "rank": 2}, corpus)
    assert "rank 2" in label and "Off-topic paper" in label and "NOT a gold paper" in label
    assert "a gold evidence paper" in _driver_label({"driver_doc": "1", "gold_docs": ["1"]}, {"doc_id": "1", "rank": 1}, corpus)
    assert _driver_label({"driver_doc": None, "gold_docs": []}, {"doc_id": "1", "rank": 1}, corpus).startswith("none (abstained)")


def test_cache_key_changes_when_a_local_checkpoint_is_retrained(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests.tiny_model import make_tiny_nli

    path = tmp_path / "ft"
    make_tiny_nli(path, seed=0)
    a = run_eval._local_model_fingerprint(str(path))
    make_tiny_nli(path, seed=1)  # "retrain" into the same folder
    assert a and run_eval._local_model_fingerprint(str(path)) != a
    assert run_eval._local_model_fingerprint("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli") is None


def test_finetune_trains_fp32_weights_from_an_fp16_checkpoint(fixture_dir, tmp_path):
    """Colab crash: the MoritzLaurer checkpoints are stored in fp16 and transformers 5 loads them as fp16."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoModelForSequenceClassification

    from eval import finetune_nli
    from tests.tiny_model import make_tiny_nli

    base = make_tiny_nli(tmp_path / "base")
    AutoModelForSequenceClassification.from_pretrained(base).half().save_pretrained(base)
    out = tmp_path / "ft"
    finetune_nli.main(["--base", base, "--data-dir", str(fixture_dir), "--out", str(out), "--epochs", "1",
                       "--max-steps", "2", "--batch-size", "4", "--device", "cpu", "--max-length", "64", "--tune-n", "3"])
    model = AutoModelForSequenceClassification.from_pretrained(out)
    assert all(p.dtype == torch.float32 for p in model.parameters())
