"""Choose a reproducible source-balanced review pool without using label hints."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .prepare import ROOT, sha256, write_jsonl
from wwii_pilot.build_shared_pool import read_jsonl


def select(candidates=ROOT / "data/cleaned_candidates.jsonl", out=ROOT / "data/selected_candidates.jsonl",
           target=10000, seed=673):
    candidates, out = Path(candidates), Path(out)
    if target < 1:
        raise ValueError("target must be positive")
    if out.exists():
        raise ValueError("selection already exists; choose a new output filename")
    rows = read_jsonl(candidates)
    if not rows or len({row['chunk_id'] for row in rows}) != len(rows):
        raise ValueError('candidates must be nonempty and have unique chunk IDs')
    bysource = defaultdict(list)
    rank = lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    for row in rows:
        bysource[row["source_id"]].append(row)
    for source in bysource:
        bysource[source].sort(key=lambda row: rank(row["chunk_id"]))
    sources = sorted(bysource, key=rank)
    selected = set()
    depth = 0
    while len(selected) < min(target, len(rows)):
        for source in sources:
            if depth < len(bysource[source]):
                selected.add(bysource[source][depth]["chunk_id"])
                if len(selected) == min(target, len(rows)):
                    break
        depth += 1
    chosen = [row for row in rows if row["chunk_id"] in selected]
    write_jsonl(out, chosen)
    counts = Counter(row["source_id"] for row in chosen)
    report = {"candidate_sha256": sha256(candidates), "selection_sha256": sha256(out),
              "candidates": len(rows), "selected": len(chosen), "target": target, "seed": seed,
              "source_articles": len(counts), "minimum_per_source": min(counts.values()),
              "maximum_per_source": max(counts.values()), "keyword_hints_used": False,
              "method": "Round-robin by article; deterministic hash order within each article.",
              "sampling_limitation": "Source-balanced sample, not a natural-frequency estimate of WWII topics."}
    out.with_suffix(".selection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=ROOT / "data/cleaned_candidates.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data/selected_candidates.jsonl")
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=673)
    args = parser.parse_args()
    print(json.dumps(select(args.candidates, args.out, args.target, args.seed), indent=2))
