"""Stage-2 reasoning: a small LOCAL LLM reads, Python does the maths.   Owner: Modeling.

Why: the claims every fine-tuned verifier gets wrong need reasoning a sentence-pair classifier cannot do,
e.g. "significantly higher" vs "P = .11", or "1/2000" vs "493 per million".  Small LLMs read such text well
but calculate badly, so the work is split:

    LocalLLM.generate()   extracts, as JSON: the sentences that bear on the claim, the claim's quantity and
                          the reported ones, p-values and confidence intervals for the claimed outcome, and
                          its own verdict.
    ground()              drops every extracted number that does not literally occur in the sentence it was
                          attributed to (a small model must not invent evidence).
    check_significance()  deterministic: is the claimed (non-)significance what the statistics say?
    check_quantity()      deterministic: do the quantities agree once units are normalised?
    combine()             safety-first: a failed check can turn the verdict into CONTRADICT; a check can only
                          CONFIRM a SUPPORT, never create one, so new false SUPPORTs can only come from the
                          LLM's own grounded verdict.

Modes (ablations): hybrid = checks + LLM verdict; checks = only the check overrides, otherwise keep stage 1;
llm = the LLM verdict alone.  Nothing here downloads anything until LocalLLM is first used.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .schema import CONTRADICT, NEI, SUPPORT

PROMPT_VERSION = "hybrid-v1"
MODES = ("hybrid", "checks", "llm")
ALPHA = 0.05

# --------------------------------------------------------------------------- #
# Numbers and units
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d*\.\d+|\d+)(?:\s*(?:/|in)\s*(\d{1,3}(?:,\d{3})+|\d+))?")
_SCALE = {"hundred": 1e2, "thousand": 1e3, "million": 1e6, "billion": 1e9}


def parse_number(x) -> Optional[float]:
    """3, "1,000", ".11", "30.9%", "1/2000", "1 in 2000", "1.5e-3" -> float (None if not a number)."""
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().rstrip("%").replace("−", "-").strip()
    m = re.fullmatch(r"(-?[\d,]*\.?\d+)\s*(?:/|in)\s*([\d,]+)", s)
    try:
        if m:
            den = float(m.group(2).replace(",", ""))
            return float(m.group(1).replace(",", "")) / den if den else None
        return float(s.replace(",", ""))
    except ValueError:
        return None


def numbers_in(text: str) -> list:
    """Every number written in the text, fractions ("1/2000", "1 in 2000") also as their value."""
    out = []
    for m in _NUM.finditer(text or ""):
        a = float(m.group(1).replace(",", ""))
        out.append(a)
        if m.group(2):
            den = float(m.group(2).replace(",", ""))
            out.append(den)
            if den:
                out.append(a / den)
    return out


def occurs_in(value, text: str) -> bool:
    v = parse_number(value)
    if v is None:
        return False
    # unsigned on purpose: "0.55-1.29" is a range, so a negative bound is matched by its magnitude
    return any(abs(abs(v) - n) <= 1e-9 * max(1.0, abs(v)) for n in numbers_in(text))


def normalize_quantity(value, unit: str) -> Optional[tuple]:
    """(value, unit) -> (kind, value): rates/percentages/"per N" become ("rate", fraction); other units are
    kept as ("unit:<name>", value) so only like is compared with like."""
    v = parse_number(value)
    if v is None:
        return None
    u = (unit or "").strip().lower()
    if isinstance(value, str) and re.search(r"\d\s*(?:/|in)\s*\d", value):
        return ("rate", v)
    if u in ("%", "percent", "percentage", "per cent") or (isinstance(value, str) and value.strip().endswith("%")):
        return ("rate", v / 100)
    m = re.fullmatch(r"(?:per|/|in)\s*([\d,\.]*)\s*(hundred|thousand|million|billion)?(?:[\s-]+[\w-]+)*", u)
    if m and (m.group(1) or m.group(2)):
        n = float(m.group(1).replace(",", "")) if m.group(1) else 1.0
        n *= _SCALE.get(m.group(2) or "", 1.0)
        return ("rate", v / n) if n else None
    if u in ("fraction", "proportion", "rate", "ratio of population", "prevalence") and 0 <= v <= 1:
        return ("rate", v)
    u = re.sub(r"s$", "", re.sub(r"\s+", " ", u))
    return (f"unit:{u}", v) if u else None


# --------------------------------------------------------------------------- #
# Deterministic checks
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    name: str
    result: Optional[str]  # SUPPORT (consistent) | CONTRADICT | None (no conclusion)
    explanation: str
    sentences: list = field(default_factory=list)


def _p_significant(p: dict) -> Optional[bool]:
    v, comp = parse_number(p.get("value")), str(p.get("comparator", "=")).strip()
    if v is None:
        return None
    if comp in ("<", "<=", "≤"):
        return True if v <= ALPHA else None  # "p < 0.2" says nothing
    if comp in (">", ">=", "≥"):
        return False if v >= ALPHA else None
    return v < ALPHA


def _ci_significant(ci: dict) -> Optional[bool]:
    lo, hi = parse_number(ci.get("low")), parse_number(ci.get("high"))
    if lo is None or hi is None or lo > hi:
        return None
    null = 1.0 if str(ci.get("measure", "ratio")).lower().startswith("ratio") else 0.0
    return not (lo <= null <= hi)


def check_significance(claim_significant, p_values: Sequence, intervals: Sequence) -> Check:
    """claim_significant: True = the claim asserts a real/significant effect, False = it asserts no effect."""
    found = [(_p_significant(p), p, f"p {p.get('comparator', '=')} {p.get('value')}") for p in p_values]
    found += [(_ci_significant(c), c, f"{c.get('measure', 'ratio')} CI {c.get('low')} to {c.get('high')}") for c in intervals]
    found = [f for f in found if f[0] is not None]
    if claim_significant is None or not found:
        return Check("significance", None, "no claimed effect or no usable statistics")
    sig = {f[0] for f in found}
    if len(sig) > 1:
        return Check("significance", None, "mixed significant and non-significant results")
    evidence_sig = sig.pop()
    desc = "; ".join(f[2] for f in found)
    sents = sorted({int(f[1]["sentence"]) for f in found if isinstance(f[1].get("sentence"), int)})
    if evidence_sig == bool(claim_significant):
        return Check("significance", SUPPORT, f"consistent with the claim ({desc})", sents)
    what = "significant" if claim_significant else "no"
    return Check("significance", CONTRADICT,
                 f"the claim asserts {what} effect, but the reported statistics say otherwise ({desc}, alpha {ALPHA})", sents)


def check_quantity(claim_q: Optional[dict], evidence_qs: Sequence) -> Check:
    if not claim_q:
        return Check("quantity", None, "the claim states no quantity")
    cq = normalize_quantity(claim_q.get("value"), claim_q.get("unit", ""))
    if cq is None:
        return Check("quantity", None, "claim quantity could not be read")
    relation = str(claim_q.get("relation", "about")).lower()
    verdicts = []
    for q in evidence_qs:
        eq = normalize_quantity(q.get("value"), q.get("unit", ""))
        if eq is None or eq[0] != cq[0]:
            continue
        c, e = cq[1], eq[1]
        if relation in ("more_than", "at_least"):
            ok, bad = e >= c, e < 0.8 * c
        elif relation in ("less_than", "at_most"):
            ok, bad = e <= c, e > 1.25 * c
        else:
            lo, hi = sorted((abs(c), abs(e)))
            ratio = hi / lo if lo else (1.0 if hi == 0 else float("inf"))
            ok, bad = ratio <= 1.25, ratio >= 2.0
        verdicts.append((SUPPORT if ok else CONTRADICT if bad else None, q, e))
    decided = [v for v in verdicts if v[0] is not None]
    if not decided:
        return Check("quantity", None, "no comparable reported quantity")
    sents = sorted({int(v[1]["sentence"]) for v in decided if isinstance(v[1].get("sentence"), int)})
    if any(v[0] == SUPPORT for v in decided):
        return Check("quantity", SUPPORT, f"a reported value matches the claimed {cq[1]:g} ({cq[0]})", sents)
    vals = ", ".join(f"{v[2]:g}" for v in decided)
    return Check("quantity", CONTRADICT,
                 f"the claim says {relation.replace('_', ' ')} {cq[1]:g} ({cq[0]}), the paper reports {vals}", sents)


# --------------------------------------------------------------------------- #
# Prompt, parsing, grounding
# --------------------------------------------------------------------------- #
_SCHEMA = """{
 "relevant_sentences": [sentence numbers that bear on the claim],
 "claim_quantity": {"value": number, "unit": "string", "relation": "about|more_than|less_than|at_least|at_most"} or null,
 "evidence_quantities": [{"value": number, "unit": "string", "sentence": number}],
 "claim_says_significant": true | false | null,
 "p_values": [{"value": number, "comparator": "<|=|>", "sentence": number}],
 "intervals": [{"low": number, "high": number, "measure": "ratio|difference", "sentence": number}],
 "verdict": "SUPPORT" | "CONTRADICT" | "NOT_ENOUGH_INFO",
 "reason": "one short sentence"
}"""

_EXAMPLE = """Example. Claim: Drug X significantly lowers mortality.
Abstract: [0] We randomised 400 patients. [1] Mortality was 12% with drug X and 14% with placebo (OR 0.84; 95% CI 0.55-1.29; P = .43).
Answer: {"relevant_sentences": [1], "claim_quantity": null, "evidence_quantities": [], "claim_says_significant": true,
"p_values": [{"value": 0.43, "comparator": "=", "sentence": 1}], "intervals": [{"low": 0.55, "high": 1.29, "measure": "ratio", "sentence": 1}],
"verdict": "CONTRADICT", "reason": "The difference in mortality was not significant."}"""


def build_prompt(claim: str, sentences: Sequence) -> str:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    return (
        "You check whether a scientific abstract supports a claim. Read carefully; do not use outside knowledge.\n"
        "Rules:\n"
        "- Copy numbers exactly as written in the abstract; do not calculate or convert anything.\n"
        "- claim_quantity: only if the claim states a specific amount, rate or proportion (keep its unit, e.g. "
        "\"%\", \"per million\", \"weeks\"). evidence_quantities: only values measuring that same thing.\n"
        "- claim_says_significant: true if the claim asserts a real or significant effect/difference/association, "
        "false if it asserts there is none, null otherwise.\n"
        "- p_values and intervals: only those for the outcome the claim is about.\n"
        "- verdict: SUPPORT or CONTRADICT only if specific sentences establish it; otherwise NOT_ENOUGH_INFO.\n"
        f"Answer with one JSON object and nothing else, in this format:\n{_SCHEMA}\n\n{_EXAMPLE}\n\n"
        f"Claim: {claim}\nAbstract:\n{numbered}\nAnswer:"
    )


def parse_json(text: str) -> Optional[dict]:
    """First balanced {...} object in the model's output (models sometimes add prose or code fences)."""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _as_list(x):
    return [i for i in x if isinstance(i, dict)] if isinstance(x, list) else []


def ground(ext: dict, claim: str, sentences: Sequence) -> tuple:
    """Keep only what the text supports.  Returns (grounded dict, number of extracted items dropped)."""
    n = len(sentences)

    def sent_ok(i):
        return isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n

    def keep(items, *value_keys):
        kept = [it for it in _as_list(items)
                if sent_ok(it.get("sentence")) and all(occurs_in(it.get(k), sentences[it["sentence"]]) for k in value_keys)]
        return kept, len(_as_list(items)) - len(kept)

    g, dropped = {}, 0
    rel = ext.get("relevant_sentences")
    g["relevant_sentences"] = sorted({i for i in rel if sent_ok(i)}) if isinstance(rel, list) else []
    cq = ext.get("claim_quantity")
    g["claim_quantity"] = cq if isinstance(cq, dict) and occurs_in(cq.get("value"), claim) else None
    dropped += int(isinstance(cq, dict) and g["claim_quantity"] is None)
    g["evidence_quantities"], d1 = keep(ext.get("evidence_quantities"), "value")
    g["p_values"], d2 = keep(ext.get("p_values"), "value")
    g["intervals"], d3 = keep(ext.get("intervals"), "low", "high")
    dropped += d1 + d2 + d3
    cs = ext.get("claim_says_significant")
    g["claim_says_significant"] = cs if isinstance(cs, bool) else None
    v = str(ext.get("verdict", "")).upper().replace(" ", "_")
    g["verdict"] = SUPPORT if v.startswith("SUPPORT") else CONTRADICT if v.startswith("CONTRADICT") else NEI
    g["reason"] = str(ext.get("reason", ""))[:300]
    return g, dropped


# --------------------------------------------------------------------------- #
# Combining
# --------------------------------------------------------------------------- #
def combine(g: dict, mode: str = "hybrid") -> dict:
    """Grounded extraction -> one paper-level judgement.

    label None means "no opinion" (keep stage 1); in llm/hybrid mode NEI is an opinion.
    """
    checks = [check_significance(g["claim_says_significant"], g["p_values"], g["intervals"]),
              check_quantity(g["claim_quantity"], g["evidence_quantities"])]
    fired = [c for c in checks if c.result == CONTRADICT]
    consistent = [c for c in checks if c.result == SUPPORT]
    info = {"checks": [c.__dict__ for c in checks], "llm_verdict": g["verdict"], "llm_reason": g["reason"]}
    if mode != "llm" and fired:
        c = fired[0]
        return {"label": CONTRADICT, "source": f"check:{c.name}", "evidence": c.sentences or g["relevant_sentences"],
                "reason": c.explanation, **info}
    if mode == "checks":
        return {"label": None, "source": "checks:none-fired", "evidence": [], "reason": "", **info}
    label, evidence = g["verdict"], g["relevant_sentences"]
    if label != NEI and not evidence:
        return {"label": NEI, "source": "llm:ungrounded", "evidence": [], "reason": "verdict without valid evidence sentences", **info}
    source = "llm" + ("+confirmed:" + ",".join(c.name for c in consistent) if label == SUPPORT and consistent and mode == "hybrid" else "")
    return {"label": label, "source": source, "evidence": evidence if label != NEI else [], "reason": g["reason"], **info}


def combine_docs(judgements: Sequence) -> dict:
    """Paper-level judgements of one claim -> claim-level stage-2 label (None = keep stage 1)."""
    opinions = [(i, j) for i, j in enumerate(judgements) if j and j.get("label") is not None]
    if not opinions:
        return {"label": None, "doc_index": None}
    verdicts = [(i, j) for i, j in opinions if j["label"] in (SUPPORT, CONTRADICT)]
    labels = {j["label"] for _, j in verdicts}
    if len(labels) > 1:
        return {"label": NEI, "doc_index": None, "note": "papers disagree"}
    if labels:
        i, _ = sorted(verdicts, key=lambda t: (not t[1]["source"].startswith("check"), t[0]))[0]
        return {"label": labels.pop(), "doc_index": i}
    return {"label": NEI, "doc_index": None}


# --------------------------------------------------------------------------- #
# The local LLM and the reasoner
# --------------------------------------------------------------------------- #
class LocalLLM:
    """A small instruction-tuned causal LM run locally with transformers (greedy, batched)."""

    def __init__(self, model_name: str, device: Optional[str] = None, max_new_tokens: int = 384, batch_size: int = 8):
        self.model_name, self.max_new_tokens, self.batch_size = model_name, max_new_tokens, batch_size
        self._device_pref, self._model = device, None

    def _load(self):
        if self._model is not None:
            return
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = self._device_pref or ("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float16 if self.device.startswith("cuda") else torch.float32
        kw = {"dtype": dt} if int(transformers.__version__.split(".")[0]) >= 5 else {"torch_dtype": dt}
        self._tok = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kw).to(self.device).eval()

    def _chat(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        try:  # enable_thinking=False: models with a "thinking" mode answer directly (others ignore it)
            return self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            return prompt

    def generate(self, prompts: Sequence) -> list:
        self._load()
        torch, out = self._torch, []
        for s in range(0, len(prompts), self.batch_size):
            texts = [self._chat(p) for p in prompts[s : s + self.batch_size]]
            enc = self._tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            with torch.no_grad():
                gen = self._model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                           pad_token_id=self._tok.pad_token_id)
            out += self._tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out


class HybridReasoner:
    def __init__(self, llm, mode: str = "hybrid"):
        if mode not in MODES:
            raise ValueError(f"unknown reasoner mode '{mode}' ({' | '.join(MODES)})")
        self.llm, self.mode = llm, mode
        self.name = f"{mode}:{getattr(llm, 'model_name', 'llm')}:{PROMPT_VERSION}"

    def extract(self, items: Sequence) -> list:
        """items: [(claim, sentences)] -> raw model outputs (the expensive, cacheable part)."""
        return self.llm.generate([build_prompt(c, s) for c, s in items])

    def judge_raw(self, claim: str, sentences: Sequence, raw: str) -> dict:
        ext = parse_json(raw)
        if ext is None:
            return {"label": None, "source": "unparseable", "evidence": [], "reason": "", "raw": raw[:500]}
        g, dropped = ground(ext, claim, sentences)
        return {**combine(g, self.mode), "dropped_ungrounded": dropped}

    def judge(self, items: Sequence) -> list:
        return [self.judge_raw(c, s, r) for (c, s), r in zip(items, self.extract(items))]
