"""Build exact, nonoverlapping short passages from the supplied Wikipedia batch."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit, urlunsplit

import nltk
from nltk.tokenize import TreebankWordTokenizer

from wwii_pilot.build_shared_pool import fingerprint, passage_id, read_jsonl


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PUBLISHED_POOL = REPO / 'data/updated_chunks'
BACKMATTER = re.compile(
    r"\b(?:references?|bibliograph(?:y|ies)|external links|see also|further reading|footnotes)\b", re.I)
DEBRIS = re.compile(r"\b(?:ISBN|ISSN|doi:|Wikimedia Commons|Retrieved\s+\d|Accessed\s+\d)\b", re.I)
UNRESOLVED = re.compile(r"^(?:He|She|They|It)\b")
WORD_TOKENIZER = TreebankWordTokenizer()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def canonical_article(url):
    """Sections and percent-encoding variants are one source article."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("source URL must be HTTP(S)")
    path = unquote(parts.path).replace(" ", "_").rstrip("/")
    return urlunsplit(("https", parts.netloc.lower(), path, "", ""))


def source_id(url):
    return "article_" + hashlib.sha256(canonical_article(url).encode()).hexdigest()[:16]


def quality_reason(text, tags, min_words=25, max_words=70, section=""):
    words = text.split()
    if not min_words <= len(words) <= max_words:
        return "word_length"
    if DEBRIS.search(text):
        return "reference_or_navigation_text"
    if UNRESOLVED.match(text):
        return "unresolved_opening_pronoun"
    if "order of battle" in section.lower() and len(re.findall(r"\b(?:Bttn|HQ|Hptm|Lt|Col|Maj|Gen)\b", text)) >= 5:
        return "order_of_battle_list"
    if sum(bool(re.fullmatch(r"[\d\W]+", word)) for word in words) > len(words) * 0.4:
        return "mostly_numbers_or_symbols"
    if not any(tag.startswith("VB") or tag == "MD" for _, tag in tags):
        return "no_verbal_clause"
    if not re.search(r"[.!?][\"'’”\])]*$", text):
        return "incomplete_ending"
    return None


def partition(text, tokenizer, tagger, min_words=25, max_words=70, section=""):
    """Choose disjoint consecutive 2/3-sentence spans, never truncate sentences."""
    spans = list(tokenizer.span_tokenize(text))
    choices = {}
    reasons = Counter()
    for start in range(len(spans)):
        for count in (2, 3):
            end = start + count
            if end > len(spans):
                continue
            begin_char, end_char = spans[start][0], spans[end - 1][1]
            excerpt = text[begin_char:end_char]
            words = len(excerpt.split())
            if not min_words <= words <= max_words:
                reasons["word_length"] += 1
                continue
            reason = quality_reason(excerpt, tagger(WORD_TOKENIZER.tokenize(excerpt)), min_words, max_words, section)
            if reason:
                reasons[reason] += 1
                continue
            choices[start, count] = (begin_char, end_char, words)

    @lru_cache(None)
    def best(i):
        if i >= len(spans):
            return (0, 0, 0), ()
        score, selected = best(i + 1)
        for count in (2, 3):
            if (i, count) not in choices:
                continue
            _, _, words = choices[i, count]
            tail, rest = best(i + count)
            candidate = (tail[0] + count, tail[1] + words, tail[2] - abs(words - 50))
            if candidate > score:
                score, selected = candidate, ((i, count),) + rest
        return score, selected

    selected = []
    for start, count in best(0)[1]:
        begin, end, _ = choices[start, count]
        selected.append({"char_start": begin, "char_end": end,
                         "sentence_start": start, "sentence_end": start + count,
                         "source_sentences": [text[a:b] for a, b in spans[start:start + count]],
                         "chunk_text": text[begin:end]})
    return selected, spans, dict(reasons)


def load_language_tools(setup=False):
    resources = ROOT / "work/nltk_data"
    resources.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(resources))
    if setup:
        for name in ("punkt_tab", "averaged_perceptron_tagger_eng"):
            if not nltk.download(name, download_dir=str(resources), quiet=True):
                raise ValueError(f"could not install NLTK resource {name}")
    try:
        tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        nltk.pos_tag(["The", "workers", "built", "factories", "."])
    except LookupError as error:
        raise ValueError("NLTK English resources missing; rerun with --setup-nltk") from error
    return tokenizer, nltk.pos_tag


def prepare(chunks, index, out=ROOT / "data", pilot=REPO / "wwii_pilot/data", setup_nltk=False):
    chunks, index, out, pilot = map(Path, (chunks, index, out, pilot))
    if out.exists():
        raise ValueError("output directory already exists; choose a new directory")
    tokenizer, tagger = load_language_tools(setup_nltk)
    parents = read_jsonl(chunks)
    incoming_index = json.loads(index.read_text(encoding="utf-8-sig"))
    anchors = json.loads((pilot / "anchors.json").read_text())
    byname = {fingerprint(a["name"]): a for a in anchors["anchors"]}
    mapping = {}
    for code, entry in incoming_index.items():
        anchor = byname.get(fingerprint(entry["name"]))
        if anchor is None:
            raise ValueError(f"unknown incoming anchor {code}: {entry['name']}")
        mapping[code] = {"code": anchor["code"], "id": anchor["id"], "name": anchor["name"]}
    if len(mapping) != len(anchors["anchors"]):
        raise ValueError("incoming anchor index does not map the entire anchor set")
    if len({r["id"] for r in parents}) != len(parents):
        raise ValueError("input has duplicate record IDs")

    prior = read_jsonl(pilot / "passages.jsonl")
    prior_text = {fingerprint(r["chunk_text"]) for r in prior}
    prior_sentences = {fingerprint(sentence) for r in prior
                       for sentence in tokenizer.tokenize(r["chunk_text"])}
    existing_sources = defaultdict(set)
    for row in prior:
        if row.get("source_id") and row.get("source_url"):
            existing_sources[canonical_article(row["source_url"])].add(row["source_id"])

    records, rejected, unused = [], [], []
    seen, sentences = set(), set()
    totals, candidate_reasons, source_counts = Counter(), Counter(), Counter()
    input_hash = sha256(chunks)
    for parent in parents:
        pid = parent["id"]
        text = parent["text"]
        if not isinstance(text, str) or not text.strip():
            rejected.append({"parent_id": pid, "reason": "empty_text"})
            continue
        page_url = canonical_article(parent["url"])
        if BACKMATTER.search(parent["section"]):
            rejected.append({"parent_id": pid, "reason": "reference_or_navigation_section"})
            continue
        pieces, spans, failure_counts = partition(text, tokenizer, tagger, section=parent["section"])
        candidate_reasons.update(failure_counts)
        used = set()
        for piece in pieces:
            key = fingerprint(piece["chunk_text"])
            sent_keys = [fingerprint(sentence) for sentence in piece["source_sentences"]]
            reason = ("already_in_pilot" if key in prior_text else
                      "duplicate_text" if key in seen else
                      "sentence_already_in_pilot" if any(k in prior_sentences for k in sent_keys) else
                      "repeated_source_sentence" if len(set(sent_keys)) != len(sent_keys)
                      or any(k in sentences for k in sent_keys) else None)
            origin = {"parent_id": pid, "char_start": piece["char_start"], "char_end": piece["char_end"],
                      "sentence_start": piece["sentence_start"], "sentence_end": piece["sentence_end"]}
            if reason:
                rejected.append({**origin, "reason": reason})
                continue
            old_ids = existing_sources[page_url]
            sid = next(iter(old_ids)) if len(old_ids) == 1 else source_id(page_url)
            records.append({
                "chunk_id": passage_id(piece["chunk_text"]), "chunk_text": piece["chunk_text"],
                "word_count": len(piece["chunk_text"].split()), "source_id": sid,
                "source_url": parent["url"], "source_page_url": page_url,
                "source_title": parent["page"], "source_section": parent["section"],
                "source_status": "provided_text_span", "source_verified": False,
                "source_license": "CC BY-SA 4.0 (Wikipedia article prose; source revision not supplied)",
                "origin": origin, "source_sentences": piece["source_sentences"],
                "raw_file_sha256": input_hash, "anchor_version": anchors["anchor_version"],
                "keyword_hints": {
                    "positive": [mapping[a["anchor"]]["code"] for a in parent["anchors"]],
                    "near_miss": [mapping[a["anchor"]]["code"] for a in parent["hard_negative_anchors"]],
                    "scope": "parent record only; not labels for this shorter passage",
                },
                "review_state": "Unreviewed", "split": None,
            })
            seen.add(key)
            sentences.update(sent_keys)
            used.update(range(piece["sentence_start"], piece["sentence_end"]))
            source_counts[sid] += 1
        for i, (start, end) in enumerate(spans):
            if i not in used:
                unused.append({"parent_id": pid, "sentence_index": i, "char_start": start, "char_end": end,
                               "reason": "not_in_retained_passage"})
        if not pieces:
            rejected.append({"parent_id": pid, "reason": "no_eligible_standalone_span",
                             "candidate_failures": failure_counts, "sentences": len(spans)})

    if not records:
        raise ValueError("no eligible passages were produced")
    out.mkdir(parents=True)
    raw = out / "raw"
    raw.mkdir()
    shutil.copyfile(chunks, raw / "chunks.jsonl")
    shutil.copyfile(index, raw / "anchor_index.json")
    assert sha256(raw / "chunks.jsonl") == input_hash
    manifest = {"chunks.jsonl": {"sha256": input_hash, "bytes": chunks.stat().st_size},
                "anchor_index.json": {"sha256": sha256(index), "bytes": index.stat().st_size}}
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "anchors.json").write_text(json.dumps(anchors, ensure_ascii=False, indent=2) + "\n")
    (out / "anchor_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n")
    write_jsonl(out / "cleaned_candidates.jsonl", records)
    write_jsonl(out / "rejections.jsonl", rejected)
    write_jsonl(out / "unused_sentences.jsonl", unused)
    report = {
        "preparation_version": 1, "nltk_version": nltk.__version__, "raw_records": len(parents),
        "cleaned_passages": len(records), "source_articles": len(source_counts),
        "sentence_counts": dict(Counter(len(r["source_sentences"]) for r in records)),
        "minimum_words": min(r["word_count"] for r in records),
        "maximum_words": max(r["word_count"] for r in records),
        "maximum_passages_per_article": max(source_counts.values()),
        "discarded_items_by_reason": dict(Counter(r["reason"] for r in rejected)),
        "candidate_window_failures": dict(candidate_reasons), "unused_sentences": len(unused),
        "all_labels_unreviewed": True, "keyword_hints_are_labels": False,
        "source_verification": "Exact spans in the supplied batch; original webpage revisions not independently verified.",
        "source_cap_policy": "No 3-per-article cap; sources must stay whole in future splits.",
        "snapshot_sha256": sha256(out / "cleaned_candidates.jsonl"),
    }
    (out / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--pilot", type=Path, default=REPO / "wwii_pilot/data")
    parser.add_argument("--setup-nltk", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.chunks, args.index, args.out, args.pilot, args.setup_nltk), indent=2))


if __name__ == "__main__":
    main()
