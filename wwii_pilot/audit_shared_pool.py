"""Read-only integrity audit for the 40-anchor working pool."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from .build_shared_pool import DEFAULT_OUT,coverage,fingerprint,passage_id,read_jsonl,near_duplicates
from .refresh_shared_pool import validate_snapshot


def audit(out=DEFAULT_OUT):
    out=Path(out);doc=json.loads((out/'anchors.json').read_text());anchors=doc['anchors'];aids={a['id'] for a in anchors}
    rows=read_jsonl(out/'passages.jsonl');errors=[];seen=set();source_counts=Counter();source_words=Counter();sentences=set()
    for r in rows:
        pid=r['chunk_id'];key=fingerprint(r['chunk_text'])
        if pid!=passage_id(r['chunk_text']):errors.append(f'{pid}: text/ID mismatch')
        if key in seen:errors.append(f'{pid}: duplicate text')
        seen.add(key)
        values=r['working_labels']
        if set(values)!=aids or any(v is not None and (type(v) is not int or v not in (0,1)) for v in values.values()):errors.append(f'{pid}: invalid labels')
        if r['review_state']!='Reviewed' and any(v is not None for v in r['labels'].values()):errors.append(f'{pid}: unapproved labels in approved-label field')
        if r.get('source_status')=='verified_exact_snapshot':
            try:validate_snapshot(r,out)
            except (ValueError,KeyError,IndexError,FileNotFoundError) as e:errors.append(f'{pid}: {e}')
            ss=r['source_sentences']
            if len(ss) not in (2,3) or len(r['chunk_text'].split())>70:errors.append(f'{pid}: new passage length')
            for s in ss:
                k=fingerprint(s)
                if k in sentences:errors.append(f'{pid}: repeated new source sentence')
                sentences.add(k)
            source_counts[r['source_id']]+=1;source_words[r['source_id']]+=len(r['chunk_text'].split())
    for sid,n in source_counts.items():
        if n>3:errors.append(f'{sid}: {n} new passages exceeds source cap')
        if source_words[sid]>200:errors.append(f'{sid}: source excerpt total exceeds 200 words')
    with (out/'labeling.csv').open(encoding='utf-8-sig',newline='') as f:edits=list(csv.DictReader(f))
    byid={r['chunk_id']:r for r in rows}
    if len(edits)!=len(rows) or {e['passage_id'] for e in edits}!=set(byid):errors.append('CSV passage IDs differ from JSONL')
    for e in edits:
        r=byid.get(e['passage_id'])
        if r is None:continue
        if e['passage_text']!=r['chunk_text'] or any(e[a] != ('' if r['working_labels'][a] is None else str(r['working_labels'][a])) for a in aids):errors.append(f'{e["passage_id"]}: CSV/JSONL values differ')
    close=near_duplicates(rows);counts=coverage(rows,anchors)
    summary=dict(passages=len(rows),anchors=len(anchors),draft_decisions=sum(v is not None for r in rows for v in r['working_labels'].values()),
                 minimum_draft_positives=min(c['draft_positive'] for c in counts),maximum_draft_positives=max(c['draft_positive'] for c in counts),
                 new_verified_excerpts=sum(source_counts.values()),new_source_groups=len(source_counts),
                 new_max_passages_per_source=max(source_counts.values(),default=0),new_max_words_per_passage=max((len(r['chunk_text'].split()) for r in rows if r.get('source_status')=='verified_exact_snapshot'),default=0),
                 new_max_excerpt_words_per_source=max(source_words.values(),default=0),near_duplicate_pairs=len(close),
                 missing_original_source=sum(not r.get('source_url') for r in rows),human_reviewed_passages=sum(r['review_state']=='Reviewed' for r in rows),
                 provisional_definitions=sum(a['definition_status']=='provisional_inferred' for a in anchors),
                 errors=errors,integrity_passed=not errors,train_ready=False,
                 limitation='Exact source verification is a text-location check, not a historical fact check or human approval of labels. Sampling quotas do not establish training adequacy.')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--report',type=Path)
    a=p.parse_args();result=audit(a.out)
    if a.report:a.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['errors']:raise SystemExit(1)
if __name__=='__main__':main()
