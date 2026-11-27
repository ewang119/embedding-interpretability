"""Corpus preparation checks using invented fixtures, not historical evidence."""
import json

from nltk.tokenize.punkt import PunktSentenceTokenizer
import pytest

from wwii_corpus import prepare as preparation


TEXT = (
    "British factories produced military equipment and essential supplies throughout the long wartime emergency. "
    "Additional workers operated the new assembly lines and increased the number of completed vehicles."
)
OTHER = (
    "American shipyards constructed cargo vessels using standardized components and coordinated production schedules. "
    "Thousands of skilled workers assembled the hull sections while government inspectors checked the completed machinery."
)


def tagger(tokens):
    verbs = {"produced", "operated", "increased", "constructed", "assembled", "checked", "worked", "created"}
    return [(token, "VBD" if token.lower() in verbs else "NN") for token in tokens]


def test_article_identity_ignores_fragments_and_url_encoding():
    a = "https://en.wikipedia.org/wiki/Example_%28History%29#Production"
    b = "https://en.wikipedia.org/wiki/Example_(History)#Workers"
    assert preparation.canonical_article(a) == preparation.canonical_article(b)
    assert preparation.source_id(a) == preparation.source_id(b)


def test_partition_preserves_exact_spans_and_sentence_sequence():
    text = TEXT.replace(". Additional", ".  Additional")
    pieces, spans, failures = preparation.partition(text, PunktSentenceTokenizer(), tagger)
    assert len(pieces) == 1
    piece = pieces[0]
    assert piece["chunk_text"] == text[piece["char_start"]:piece["char_end"]] == text
    assert piece["source_sentences"] == [text[a:b] for a, b in spans]
    assert piece["sentence_end"] - piece["sentence_start"] == 2


def test_overlong_sentence_is_skipped_without_truncating_it():
    text = " ".join(["Lengthy"] * 75) + ". " + TEXT
    pieces, spans, failures = preparation.partition(text, PunktSentenceTokenizer(), tagger)
    assert len(pieces) == 1
    assert pieces[0]["chunk_text"] == TEXT
    assert pieces[0]["sentence_start"] == 1


def test_three_sentences_can_prevent_an_orphan():
    text = TEXT + " New teams worked through the evening to complete the remaining government orders."
    pieces, _, _ = preparation.partition(text, PunktSentenceTokenizer(), tagger)
    assert len(pieces) == 1
    assert len(pieces[0]["source_sentences"]) == 3


def test_partition_never_reuses_a_sentence():
    pieces, _, _ = preparation.partition(TEXT + " " + OTHER, PunktSentenceTokenizer(), tagger)
    covered = []
    for piece in pieces:
        covered.extend(range(piece["sentence_start"], piece["sentence_end"]))
        assert 25 <= len(piece["chunk_text"].split()) <= 70
    assert len(covered) == len(set(covered)) == 4


def test_reference_fragments_and_anonymous_openings_are_rejected():
    assert preparation.quality_reason(TEXT + " ISBN 12345.", tagger(TEXT.split())) == "reference_or_navigation_text"
    assert preparation.quality_reason("He " + TEXT, tagger(TEXT.split())) == "unresolved_opening_pronoun"
    assert preparation.quality_reason(TEXT, [(word, "NN") for word in TEXT.split()]) == "no_verbal_clause"


def fixture_files(tmp_path):
    chunks, index, pilot = tmp_path / "chunks.jsonl", tmp_path / "index.json", tmp_path / "pilot"
    pilot.mkdir()
    anchors = {"anchor_version": "fixture-v1", "anchors": [
        {"id": "industry", "code": "A01", "name": "Industry", "definition_status": "user_supplied"}
    ]}
    (pilot / "anchors.json").write_text(json.dumps(anchors))
    (pilot / "passages.jsonl").write_text("")
    index.write_text(json.dumps({"MOB": {"name": "Industry", "positive": ["r1"], "hard_negative": [], "easy_negative": []}}))
    rows = [
        {"id": "r1", "page": "Example", "url": "https://example.org/article#One", "section": "Production",
         "text": TEXT, "anchors": [{"anchor": "MOB"}], "hard_negative_anchors": []},
        {"id": "r2", "page": "Example", "url": "https://example.org/article#Two", "section": "Workers",
         "text": OTHER, "anchors": [], "hard_negative_anchors": [{"anchor": "MOB"}]},
        {"id": "r3", "page": "Example", "url": "https://example.org/article#References", "section": "References",
         "text": TEXT, "anchors": [], "hard_negative_anchors": []},
    ]
    chunks.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return chunks, index, pilot, rows


def test_prepare_preserves_inputs_and_keeps_keyword_hints_out_of_labels(tmp_path, monkeypatch):
    chunks, index, pilot, parents = fixture_files(tmp_path)
    monkeypatch.setattr(preparation, "load_language_tools", lambda _: (PunktSentenceTokenizer(), tagger))
    before = chunks.read_bytes(), index.read_bytes()
    out = tmp_path / "corpus"
    report = preparation.prepare(chunks, index, out, pilot)
    rows = preparation.read_jsonl(out / "cleaned_candidates.jsonl")
    assert report["cleaned_passages"] == 2
    assert report["source_articles"] == 1
    assert all(row["review_state"] == "Unreviewed" for row in rows)
    assert all("labels" not in row and "working_labels" not in row for row in rows)
    assert rows[0]["keyword_hints"]["positive"] == ["A01"]
    assert rows[1]["keyword_hints"]["near_miss"] == ["A01"]
    assert (chunks.read_bytes(), index.read_bytes()) == before
    assert (out / "raw/chunks.jsonl").read_bytes() == before[0]
    assert (out / "raw/anchor_index.json").read_bytes() == before[1]
    for row in rows:
        origin = row["origin"]
        parent = next(p for p in parents if p["id"] == origin["parent_id"])
        assert row["chunk_text"] == parent["text"][origin["char_start"]:origin["char_end"]]
    rejected = preparation.read_jsonl(out / "rejections.jsonl")
    assert {r["reason"] for r in rejected} == {"reference_or_navigation_section"}


def test_repeated_sentences_are_not_retained_in_different_passages(tmp_path, monkeypatch):
    chunks, index, pilot, parents = fixture_files(tmp_path)
    parents[1]["text"] = TEXT
    chunks.write_text("".join(json.dumps(row) + "\n" for row in parents))
    monkeypatch.setattr(preparation, "load_language_tools", lambda _: (PunktSentenceTokenizer(), tagger))
    report = preparation.prepare(chunks, index, tmp_path / "out", pilot)
    assert report["cleaned_passages"] == 1
    assert report["discarded_items_by_reason"]["duplicate_text"] == 1


def test_existing_output_is_not_overwritten(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    (out / "marker").write_text("preserve")
    with pytest.raises(ValueError, match="already exists"):
        preparation.prepare(tmp_path / "absent", tmp_path / "index", out)
    assert (out / "marker").read_text() == "preserve"
