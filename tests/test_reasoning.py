"""Stage-2 hybrid reasoner: deterministic checks, grounding, combining, and the run_eval cascade (fake LLM)."""
import json

import pytest

from eval import run_eval
from src import reasoning as R
from src.schema import CONTRADICT, NEI, SUPPORT
from tests.test_run_eval import base_args

NICOTINE = ("At 52 weeks, 30.9% of the combination group were abstinent compared with 24.5% with varenicline "
            "(OR, 1.39; 95% CI, 0.93-2.07; P = .11).")
PRP = ("16 were positive for abnormal PrP, indicating an overall prevalence of 493 per million population "
       "(95% confidence interval 282 to 801 per million).")


@pytest.mark.parametrize("value,expected", [(3, 3.0), ("1,000", 1000.0), (".11", 0.11), ("30.9%", 30.9),
                                            ("1/2000", 0.0005), ("1 in 2000", 0.0005), ("abc", None), (True, None)])
def test_parse_number(value, expected):
    assert R.parse_number(value) == expected


def test_normalize_quantity_puts_rates_on_one_scale():
    assert R.normalize_quantity("1/2000", "") == ("rate", 0.0005)
    assert R.normalize_quantity(493, "per million") == ("rate", 0.000493)
    assert R.normalize_quantity(5, "per 100,000 person-years") == ("rate", 5e-05)
    assert R.normalize_quantity(30.9, "%") == ("rate", 0.309)
    assert R.normalize_quantity(12, "weeks") == ("unit:week", 12.0)
    assert R.normalize_quantity(1000, "") is None  # a bare count cannot be compared with anything


def test_significance_check():
    nonsig = [{"value": 0.11, "comparator": "=", "sentence": 0}]
    assert R.check_significance(True, nonsig, []).result == CONTRADICT
    assert R.check_significance(False, nonsig, []).result == SUPPORT
    assert R.check_significance(True, [{"value": 0.001, "comparator": "<", "sentence": 0}], []).result == SUPPORT
    assert R.check_significance(True, [], [{"low": 0.93, "high": 2.07, "measure": "ratio"}]).result == CONTRADICT
    assert R.check_significance(True, [], [{"low": 0.2, "high": 0.9, "measure": "difference"}]).result == SUPPORT
    assert R.check_significance(True, [{"value": 0.2, "comparator": "<"}], []).result is None   # "p < 0.2" is no evidence
    assert R.check_significance(None, nonsig, []).result is None
    mixed = nonsig + [{"value": 0.01, "comparator": "=", "sentence": 0}]
    assert R.check_significance(True, mixed, []).result is None


def test_quantity_check():
    claim = {"value": "1/2000", "unit": "", "relation": "about"}
    assert R.check_quantity(claim, [{"value": 493, "unit": "per million", "sentence": 0}]).result == SUPPORT
    assert R.check_quantity(claim, [{"value": 100, "unit": "per million", "sentence": 0}]).result == CONTRADICT
    assert R.check_quantity(claim, [{"value": 800, "unit": "per million", "sentence": 0}]).result is None  # 1.6x: unsure
    assert R.check_quantity(claim, [{"value": 12, "unit": "weeks", "sentence": 0}]).result is None        # unlike units
    more = {"value": 50, "unit": "%", "relation": "more_than"}
    assert R.check_quantity(more, [{"value": 60, "unit": "%"}]).result == SUPPORT
    assert R.check_quantity(more, [{"value": 30, "unit": "%"}]).result == CONTRADICT


def test_grounding_drops_invented_numbers_and_bad_sentence_ids():
    ext = {"relevant_sentences": [0, 7, "x"], "claim_quantity": {"value": 5, "unit": "%"},
           "evidence_quantities": [{"value": 493, "unit": "per million", "sentence": 0},
                                   {"value": 999, "unit": "per million", "sentence": 0},
                                   {"value": 493, "unit": "per million", "sentence": 3}],
           "p_values": [{"value": 0.11, "comparator": "=", "sentence": 0}],
           "intervals": [{"low": 282, "high": 801, "measure": "ratio", "sentence": 0}],
           "claim_says_significant": "yes", "verdict": "supports"}
    g, dropped = R.ground(ext, "Prevalence is 1/2000.", [PRP])
    assert g["relevant_sentences"] == [0]
    assert g["claim_quantity"] is None                        # 5 is not in the claim
    assert g["evidence_quantities"] == [ext["evidence_quantities"][0]]
    assert g["p_values"] == [] and len(g["intervals"]) == 1   # 0.11 is not in this sentence
    assert g["claim_says_significant"] is None and g["verdict"] == SUPPORT
    assert dropped == 4


def test_combine_is_safety_first():
    base = {"relevant_sentences": [0], "claim_quantity": None, "evidence_quantities": [], "claim_says_significant": True,
            "p_values": [{"value": 0.11, "comparator": "=", "sentence": 0}], "intervals": [], "reason": "r"}
    # a failed check overrides the LLM's SUPPORT (hybrid and checks), but not in llm-only mode
    for mode, label in (("hybrid", CONTRADICT), ("checks", CONTRADICT), ("llm", SUPPORT)):
        assert R.combine({**base, "verdict": SUPPORT}, mode)["label"] == label
    # a consistent check never creates a SUPPORT on its own
    ok = {**base, "p_values": [{"value": 0.001, "comparator": "<", "sentence": 0}]}
    assert R.combine({**ok, "verdict": NEI}, "hybrid")["label"] == NEI
    assert R.combine({**ok, "verdict": SUPPORT}, "hybrid")["source"] == "llm+confirmed:significance"
    assert R.combine({**ok, "verdict": SUPPORT}, "checks")["label"] is None   # checks mode: keep stage 1
    # a verdict with no valid evidence sentence is not accepted
    assert R.combine({**ok, "verdict": SUPPORT, "relevant_sentences": [], "p_values": []}, "hybrid")["source"] == "llm:ungrounded"


def test_combine_docs():
    j = lambda label, source="llm": {"label": label, "source": source}  # noqa: E731
    assert R.combine_docs([j(None), j(NEI)]) == {"label": NEI, "doc_index": None}
    assert R.combine_docs([j(NEI), j(SUPPORT)])["label"] == SUPPORT
    assert R.combine_docs([j(SUPPORT), j(CONTRADICT)])["label"] == NEI
    assert R.combine_docs([j(CONTRADICT), j(CONTRADICT, "check:significance")])["doc_index"] == 1
    assert R.combine_docs([j(None)])["label"] is None


def test_parse_json_tolerates_chatter():
    assert R.parse_json('Sure:\n```json\n{"verdict": "SUPPORT", "reason": "a {b} c"}\n```') == {"verdict": "SUPPORT", "reason": "a {b} c"}
    assert R.parse_json("no json here") is None
    assert R.parse_json('{"broken": } {"ok": 1}') == {"ok": 1}


def test_reasoner_end_to_end_with_a_scripted_llm():
    class Scripted:
        model_name = "scripted"

        def generate(self, prompts):
            assert all("[0] At 52 weeks" in p and "Claim: Combination" in p for p in prompts)
            return [json.dumps({"relevant_sentences": [0], "claim_says_significant": True,
                                "p_values": [{"value": 0.11, "comparator": "=", "sentence": 0}],
                                "verdict": "SUPPORT", "reason": "higher abstinence"})]

    out = R.HybridReasoner(Scripted()).judge([("Combination therapy gives significantly higher abstinence.", [NICOTINE])])[0]
    assert out["label"] == CONTRADICT and out["source"] == "check:significance" and out["evidence"] == [0]
    assert "0.11" in out["reason"]
    with pytest.raises(ValueError):
        R.HybridReasoner(Scripted(), mode="vote")


class FakeLLM:
    """Says SUPPORT, citing sentence 0, for every paper; counts calls."""
    calls = 0

    def __init__(self, model_name, **_):
        self.model_name = model_name

    def generate(self, prompts):
        FakeLLM.calls += len(prompts)
        return [json.dumps({"relevant_sentences": [0], "verdict": "SUPPORT", "reason": "fake"}) for _ in prompts]


def test_cascade_in_run_eval(fixture_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "LocalLLM", FakeLLM)
    monkeypatch.chdir(tmp_path)  # the reasoner cache lives in ./eval/cache
    extra = ["--reasoner", "fake-llm", "--escalate", "all", "--oracle"]
    r = run_eval.main(base_args(fixture_dir, tmp_path / "res", "s2", extra=extra))
    st = r["stage2"]["runs"]["end_to_end"]
    assert st["n_escalated"] > 0 and st["n_llm_calls"] == st["n_paper_judgements"] > 0
    assert r["stage2"]["runs"]["oracle_doc"]["n_escalated"] > 0 and "cited_paper" in r
    preds = [json.loads(l) for l in (tmp_path / "res" / "s2" / "predictions.jsonl").read_text().splitlines()]
    escalated = [p for p in preds if "stage2" in p]
    assert escalated and all(p["pred"] == SUPPORT and p["note"].startswith("stage2:llm") for p in escalated)
    assert "stage2:hybrid" in (tmp_path / "res" / "LATEST.md").read_text()
    assert "Stage 2" in (tmp_path / "res" / "s2" / "summary.md").read_text()

    # another mode of the same model reuses the cached generations: no new LLM calls
    before = FakeLLM.calls
    cached = lambda name, more: [a for a in base_args(fixture_dir, tmp_path / "res", name, extra=more) if a != "--no-cache"]  # noqa: E731
    run_eval.main(cached("s2w", extra))  # warm the cache
    assert FakeLLM.calls > before
    before = FakeLLM.calls
    r2 = run_eval.main(cached("s2c", extra + ["--reasoner-mode", "checks"]))
    assert FakeLLM.calls == before and r2["stage2"]["runs"]["end_to_end"]["n_llm_calls"] == 0
    assert r2["claim_level"] == run_eval.main(base_args(fixture_dir, tmp_path / "res", "base", extra=["--oracle"]))["claim_level"]

    with pytest.raises(SystemExit):
        run_eval.main(base_args(fixture_dir, tmp_path / "res", "p", extra=["--reasoner", "fake-llm", "--promote"]))


def test_not_confident_policy():
    out = {"docs": [{"probs": {"support": 0.99, "neutral": 0.0, "contradict": 0.01}}]}
    assert not run_eval.needs_reasoning({"abstained": False}, out, "not-confident")
    assert run_eval.needs_reasoning({"abstained": True}, out, "not-confident")
    out["docs"][0]["probs"]["support"] = 0.9
    assert run_eval.needs_reasoning({"abstained": False}, out, "not-confident")
    assert not run_eval.needs_reasoning({"abstained": True}, {"docs": []}, "all")


def test_local_llm_runs_a_real_causal_lm(tmp_path):
    """Plumbing only: a tiny random GPT-2 produces gibberish, which must come back as 'no opinion'."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import BertTokenizerFast, GPT2Config, GPT2LMHeadModel

    from tests.tiny_model import make_tiny_nli

    tok_dir = make_tiny_nli(tmp_path / "tok")
    tok = BertTokenizerFast.from_pretrained(tok_dir)
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(vocab_size=tok.vocab_size, n_positions=1024, n_embd=32, n_layer=2, n_head=2)).save_pretrained(tmp_path / "lm")
    tok.save_pretrained(tmp_path / "lm")
    llm = R.LocalLLM(str(tmp_path / "lm"), device="cpu", max_new_tokens=8, batch_size=2)
    outs = R.HybridReasoner(llm).judge([("Zorvex helps bones.", ["Zorvex helps bones in mice."])] * 3)
    assert len(outs) == 3 and all(o["label"] is None and o["source"] == "unparseable" for o in outs)
