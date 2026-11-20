"""Offline evaluation: retrieval quality, answer quality and safety in one run.

    python -m bluewhale.evals.run                       # default dataset, hybrid
    python -m bluewhale.evals.run --mode dense --k 3    # compare retrieval modes
    python -m bluewhale.evals.run --prompt-version 1    # compare prompt versions
    python -m bluewhale.evals.run --gate                # exit 1 if below thresholds (CI)

Metrics
* recall@k   share of answerable questions whose source doc is in the top k
* MRR        mean reciprocal rank of the first correct doc
* answer_hit share of answerable questions whose answer contains the expected fact
* refusal    share of unanswerable / adversarial questions correctly refused
* grounding  mean supported-term ratio of answered questions
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import Settings
from ..container import build_services, load_samples

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "data" / "evals" / "qa.jsonl"
GATES = {"recall_at_k": 0.85, "mrr": 0.6, "answer_hit": 0.75, "refusal": 1.0}


@dataclass
class EvalReport:
    mode: str
    k: int
    prompt_version: int | None
    questions: int
    recall_at_k: float
    mrr: float
    answer_hit: float
    refusal: float
    grounding: float
    total_input_tokens: int
    failures: list[dict]


def run_eval(dataset: Path = DEFAULT_DATASET, mode: str = "hybrid", k: int = 5,
             prompt_version: int | None = None) -> EvalReport:
    services = build_services(Settings())
    load_samples(services.rag)
    services.rag.prompt_version = prompt_version
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]

    hits = rr = answered_ok = refused_ok = 0
    answerable = unanswerable = 0
    grounding: list[float] = []
    tokens_in = 0
    failures: list[dict] = []

    for row in rows:
        q = row["question"]
        expected = set(row.get("expected_docs", []))
        if expected:
            answerable += 1
            results = services.rag.retriever.retrieve(q, k=k, mode=mode)
            docs = list(dict.fromkeys(r.chunk.doc_id for r in results))
            rank = next((i for i, d in enumerate(docs) if d in expected), None)
            if rank is not None:
                hits += 1
                rr += 1.0 / (rank + 1)
            else:
                failures.append({"id": row["id"], "type": "retrieval", "got": docs[:k]})

        ans = services.rag.answer(q)
        tokens_in += int(ans.usage.get("input_tokens", 0))
        if row.get("expect_refusal"):
            unanswerable += 1
            if ans.refused or "could not find" in ans.answer.lower():
                refused_ok += 1
            else:
                failures.append({"id": row["id"], "type": "should_refuse", "answer": ans.answer})
            continue
        if not ans.refused:
            grounding.append(ans.grounding_score)
        needles = row.get("must_contain", [])
        if needles and all(n.lower() in ans.answer.lower() for n in needles):
            answered_ok += 1
        else:
            failures.append({"id": row["id"], "type": "answer", "answer": ans.answer[:200]})

    def ratio(a: float, b: int) -> float:
        return round(a / b, 3) if b else 1.0

    return EvalReport(mode, k, prompt_version, len(rows), ratio(hits, answerable),
                      ratio(rr, answerable), ratio(answered_ok, answerable),
                      ratio(refused_ok, unanswerable),
                      round(sum(grounding) / len(grounding), 3) if grounding else 0.0,
                      tokens_in, failures)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--mode", choices=["hybrid", "dense", "keyword"], default="hybrid")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--prompt-version", type=int, default=None)
    p.add_argument("--gate", action="store_true", help="exit non-zero below thresholds")
    p.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = p.parse_args(argv)

    report = run_eval(args.dataset, args.mode, args.k, args.prompt_version)
    payload = json.dumps(asdict(report), indent=2)
    print(payload)
    if args.out:
        args.out.write_text(payload)
    if args.gate:
        below = {m: getattr(report, m) for m, t in GATES.items() if getattr(report, m) < t}
        if below:
            print(f"eval gate failed: {below}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
