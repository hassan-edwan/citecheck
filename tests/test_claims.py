"""Citation audit: all contexts, LLM claim extraction (scripted), grounding, faithfulness, aggregation."""
import json

from src.claims import ClaimExtractor, aggregate, build_prompt, parse_claims
from src.pipeline import CiteCheckPipeline
from src.schema import CONTRADICT, NEI, PARTIAL, Paper, Probs, Reference, SUPPORT
from tests.test_pipeline import Scripted

CTX = "Zorvex increases bone density [3] but causes headaches, unlike our drug."
PAPER = Paper("p", "Zorvex trial", "Zorvex increased bone density in mice. No adverse events were recorded.")


def llm_saying(*outputs):
    class LLM:
        model_name = "scripted"
        prompts = []

        def generate(self, prompts):
            LLM.prompts += prompts
            return [outputs[i % len(outputs)] for i in range(len(prompts))]

    return LLM()


def claims_json(*claims):
    return json.dumps({"claims": [dict(zip(("claim", "source_quote", "role", "needs_more_context"), c)) for c in claims]})


class ByClaim:
    """Verifier keyed on a word in the CLAIM (every claim here is checked against the same abstract)."""

    def __init__(self, table, default=Probs(0.05, 0.9, 0.05)):
        self.table, self.default = table, default

    def predict_pairs(self, pairs):
        return [next((p for w, p in self.table.items() if w in claim.lower()), self.default) for _, claim in pairs]


class Entails:
    """Faithfulness verifier: entailment iff the two texts share their first word (enough to script both cases)."""

    def predict_pairs(self, pairs):
        return [Probs(0.9, 0.05, 0.05) if a.split()[0].lower() == b.split()[0].lower() else Probs(0.1, 0.8, 0.1) for a, b in pairs]


def test_aggregate():
    assert aggregate([SUPPORT, SUPPORT]) == SUPPORT
    assert aggregate([SUPPORT, NEI]) == PARTIAL
    assert aggregate([SUPPORT, CONTRADICT]) == CONTRADICT
    assert aggregate([NEI, NEI]) == NEI
    assert aggregate([None]) is None


def test_prompt_names_the_cited_paper_and_intents():
    p = build_prompt(CTX, "Zorvex trial", ("result",))
    assert "Zorvex trial" in p and "intent as: result" in p and CTX in p


def test_parse_claims_grounds_quotes_and_flags_scope():
    raw = "Here you go: " + claims_json(
        ("Zorvex increases bone density.", "Zorvex increases bone density", "result", False),
        ("Zorvex causes headaches.", "zorvex causes migraines", "result", False),      # quote not in sentence
        ("It is safe.", "causes headaches", "result", True),                           # ambiguous
        ("Zorvex is a drug.", "Zorvex", "weird-role", False))
    out = parse_claims(raw, CTX)
    assert [c.status for c in out] == ["checked", "ungrounded", "needs_more_context", "checked"]
    assert out[3].role == "other"
    assert parse_claims("no json", CTX) is None and parse_claims('{"claims": "x"}', CTX) is None


def test_faithfulness_rejects_meaning_changing_rewrites():
    ex = ClaimExtractor(llm_saying(claims_json(
        ("Zorvex increases bone density.", "Zorvex increases bone density", "result", False),
        ("Bone loss is reversed by zorvex.", "Zorvex increases bone density", "result", False))), verifier=Entails())
    a, b = ex.extract([(CTX, "Zorvex trial", ())])[0]
    assert a.status == "checked" and a.faithfulness == (0.9, 0.9)
    assert b.status == "unfaithful" and "does not mean the same" in b.note


def test_audit_checks_every_context_without_an_extractor():
    pipe = CiteCheckPipeline(ByClaim({"bone density": Probs(0.9, 0.05, 0.05)}), tau=0.5)
    ref = Reference(PAPER, context="Zorvex raises bone density [3].",
                    contexts=("Zorvex raises bone density [3].", "Zorvex is used in elderly patients [3]."))
    row = pipe.audit([ref])[0]
    assert len(row.claims) == 2 and row.verdict.label == PARTIAL and row.verdict.note == "1 of 2 claims supported"
    assert row.verdict.display_label == "PARTLY SUPPORTED"
    assert ref.all_contexts == ref.contexts and Reference(PAPER, context="x").all_contexts == ("x",)


def test_audit_with_extractor_checks_only_attributed_claims():
    verifier = ByClaim({"bone density": Probs(0.9, 0.05, 0.05), "headaches": Probs(0.05, 0.1, 0.85)})
    pipe = CiteCheckPipeline(verifier, tau=0.5)
    pipe.claim_extractor = ClaimExtractor(llm_saying(claims_json(
        ("Zorvex increases bone density.", "Zorvex increases bone density", "result", False),
        ("Zorvex causes headaches.", "causes headaches", "result", False))))
    row = pipe.audit([Reference(PAPER, context=CTX)])[0]
    assert [c.claim for c in row.claims] == ["Zorvex increases bone density.", "Zorvex causes headaches."]
    assert [c.verdict.label for c in row.claims] == [SUPPORT, CONTRADICT] and row.verdict.label == CONTRADICT

    # nothing attributed (method-only citation): skipped, not judged
    pipe.claim_extractor = ClaimExtractor(llm_saying('{"claims": []}'))
    row = pipe.audit([Reference(PAPER, context="We used the protocol of [3].")])[0]
    assert row.verdict is None and row.claims[0].status == "not_attributed" and "could be checked" in row.skipped_reason

    # unusable LLM output: fall back to the whole sentence, and say so
    pipe.claim_extractor = ClaimExtractor(llm_saying("sorry, I cannot help"))
    row = pipe.audit([Reference(PAPER, context="Zorvex raises bone density [3].")])[0]
    assert row.verdict.label == SUPPORT and "extraction failed" in row.claims[0].note


def test_audit_survives_a_crashing_llm():
    class Broken:
        model_name = "broken"

        def generate(self, prompts):
            raise RuntimeError("CUDA out of memory")

    pipe = CiteCheckPipeline(Scripted({"bone density": Probs(0.9, 0.05, 0.05)}), tau=0.5)
    pipe.claim_extractor = ClaimExtractor(Broken())
    row = pipe.audit([Reference(PAPER, context="Zorvex raises bone density [3].")])[0]
    assert row.verdict.label == SUPPORT and any("Claim extraction unavailable" in w for w in pipe.warnings)


def test_build_pipeline_attaches_extractor_from_env(monkeypatch, fixture_dir):
    from src.pipeline import build_pipeline

    monkeypatch.setenv("CITECHECK_VERIFIER", "lexical")
    monkeypatch.setenv("CITECHECK_DATA_DIR", str(fixture_dir))
    monkeypatch.setenv("CITECHECK_CLAIM_EXTRACTOR", "some/local-llm")
    pipe = build_pipeline()
    assert pipe.claim_extractor is not None and pipe.meta["claim_extractor"] == "some/local-llm"


def test_audit_gate_and_audit_tau(monkeypatch, fixture_dir):
    pipe = CiteCheckPipeline(ByClaim({"bone density": Probs(0.7, 0.2, 0.1)}), tau=0.5)
    method = Reference(PAPER, context="Cells were cultured as described previously [3].")
    real = Reference(PAPER, context="Zorvex raises bone density [3].")
    rows = pipe.audit([method, real])
    assert rows[0].verdict is None and rows[0].claims[0].status == "not_a_claim" and "method" in rows[0].claims[0].note
    assert rows[1].verdict.label == SUPPORT
    pipe.audit_tau = 0.8                                     # the audit's own threshold, stricter than the checker's
    assert pipe.audit([real])[0].verdict.label == NEI
    assert pipe.verify_against_paper("Zorvex raises bone density.", PAPER).label == SUPPORT  # claim checker unchanged
    pipe.citation_gate = False
    assert pipe.audit([method])[0].verdict is not None       # gate off: the method citation gets judged

    from src.config import load_config
    from src.pipeline import build_pipeline

    monkeypatch.setenv("CITECHECK_VERIFIER", "lexical")
    monkeypatch.setenv("CITECHECK_DATA_DIR", str(fixture_dir))
    monkeypatch.setenv("CITECHECK_AUDIT_TAU", "0.75")
    monkeypatch.setenv("CITECHECK_CITATION_GATE", "off")
    p = build_pipeline()
    assert p.audit_tau == 0.75 and p.citation_gate is False and load_config().audit_tau == 0.75
