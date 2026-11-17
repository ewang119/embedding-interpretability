"""Refresh counts from edited labels and append explicitly reviewed new passages."""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import zipfile
import xml.etree.ElementTree as ET
from io import StringIO
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from .build_shared_pool import DEFAULT_OUT, fingerprint, passage_id, read_jsonl, near_duplicates, write_outputs


def parse_label(value):
    if value in ('', None):
        return None
    if value in ('0','1'):
        return int(value)
    raise ValueError(f'Labels must be 0, 1, or blank; got {value!r}')


def apply_csv(rows, path, anchors):
    """An empty cell remains unreviewed; text edits require a new passage ID."""
    original={r['chunk_id']:r for r in rows}
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        edits=list(csv.DictReader(f))
    seen=set()
    for e in edits:
        pid=e['passage_id']
        if not pid:
            if any(v for v in e.values()):
                raise ValueError('Nonempty row has no passage_id')
            continue
        if pid in seen or pid not in original:
            raise ValueError(f'Duplicate or unknown passage_id: {pid}')
        seen.add(pid);r=original[pid]
        if e['passage_text']!=r['chunk_text']:
            raise ValueError(f'Text changed for {pid}; add a new passage instead of changing its identity')
        labels={a['id']:parse_label(e[a['id']]) for a in anchors}
        state=e['review_state']
        if state not in {'AI draft','Needs review','Reviewed','Unreviewed','Conflict'}:
            raise ValueError(f'Invalid review state: {state}')
        if state=='Reviewed' and None in labels.values():
            raise ValueError(f'Reviewed passage has blank labels: {pid}')
        r['working_labels']=labels;r['review_state']=state
        r['labels']=dict(labels) if state=='Reviewed' else {a['id']:None for a in anchors}
        if e['source_verified'] not in ('yes','no'):
            raise ValueError('source_verified must be yes or no')
        r.update(source_id=e['source_id'] or None,source_url=e['source_url'] or None,
                 source_verified=e['source_verified']=='yes')
        if r['source_verified'] and not (r['source_id'] and r['source_url']):
            raise ValueError(f'Verified source lacks ID/URL: {pid}')
    if seen!=set(original):
        raise ValueError('CSV omitted existing passages; deletion must be explicit')
    return rows


def read_xlsx_labels(path, anchors):
    """Read the editable Labels sheet without treating blanks as numeric zero."""
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
        'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
    with zipfile.ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(t.text or '' for t in si.findall('.//m:t',ns)) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
        book=ET.fromstring(z.read('xl/workbook.xml'))
        sheet=next((e for e in book.findall('m:sheets/m:sheet',ns) if e.get('name')=='Labels'),None)
        if sheet is None: raise ValueError('Workbook has no Labels sheet')
        rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        target=next(e.get('Target') for e in rels if e.get('Id')==sheet.get('{'+ns['r']+'}id'))
        member=target.lstrip('/') if target.startswith('/') else 'xl/'+target
        xml=ET.fromstring(z.read(member))
        rows=[]
        for row in xml.findall('m:sheetData/m:row',ns):
            cells={}
            for c in row.findall('m:c',ns):
                ref=c.get('r');col=''.join(x for x in ref if x.isalpha())
                if c.find('m:f',ns) is not None: raise ValueError('Labels sheet contains a formula; use explicit values')
                v=c.find('m:v',ns);raw=v.text if v is not None and v.text else ''
                if c.get('t')=='b':raise ValueError('Use numeric 0/1 labels instead of Boolean cells')
                if c.get('t')=='s':raw=strings[int(raw)] if raw else ''
                elif c.get('t')=='inlineStr':raw=''.join(t.text or '' for t in c.findall('.//m:t',ns))
                elif c.get('t') not in ('str','e','b') and raw in ('0.0','1.0'):raw=raw[0]
                cells[col]=raw
            rows.append(cells)
    if not rows: raise ValueError('Labels sheet is empty')
    headers=rows[0]
    code_map={a['code']:a['id'] for a in anchors}
    fields=[code_map.get(h,h) for h in headers.values()]
    expected={'passage_id','passage_text','review_state','source_id','source_url','source_verified',*[a['id'] for a in anchors]}
    if set(fields)!=expected or len(fields)!=len(expected):raise ValueError('Unexpected or duplicate workbook columns')
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for cells in rows[1:]:
        record={code_map.get(h,h):cells.get(c,'') for c,h in headers.items()}
        if not record['passage_id'] and not any(record.values()):continue
        writer.writerow(record)
    return stream.getvalue()


def validate_snapshot(record, root):
    """Match a new excerpt to its saved source paragraph before marking it verified."""
    root=Path(root).resolve();path=(root/record['source_snapshot']).resolve()
    if root not in path.parents:
        raise ValueError('Source snapshot must be inside the dataset directory')
    doc=json.loads(path.read_text())
    if record['source_html_sha256']!=doc['sha256']:
        raise ValueError('Source HTML hash does not match the saved snapshot')
    pi,start,end=(record[k] for k in ('paragraph_index','char_start','char_end'))
    if any(type(x) is not int for x in (pi,start,end)) or pi<0 or start<0 or end<=start:
        raise ValueError('Invalid source text offsets')
    par=doc['paragraphs'][pi]
    if par[start:end]!=record['chunk_text']:
        raise ValueError('Excerpt does not exactly match the saved source location')
    if ' '.join(record['source_sentences'])!=record['chunk_text']:
        raise ValueError('Source sentence sequence does not match the excerpt')
    return True


def append_reviewed(rows, additions, anchors, version):
    keys={fingerprint(r['chunk_text']) for r in rows}
    counts=Counter(r['source_id'] for r in rows if r.get('source_id'))
    sentences=set()
    valid=set(range(1,len(anchors)+1))
    for r in rows:
        for s in r.get('source_sentences',[]): sentences.add(fingerprint(s))
    for a in additions:
        key=fingerprint(a['chunk_text'])
        if key in keys: raise ValueError('Duplicate passage in additions')
        if not a.get('source_id') or not a.get('source_url'):
            raise ValueError('New collection requires a source ID and URL')
        if counts[a['source_id']]>=3:
            raise ValueError(f'More than 3 passages from {a["source_id"]}')
        positives=a['positive_anchor_numbers']
        if any(type(x) is not int for x in positives) or len(positives)!=len(set(positives)) or not set(positives)<=valid:
            raise ValueError('Invalid positive anchor IDs')
        if a.get('review_method')!='assistant_semantic_review_all_anchors':
            raise ValueError('An explicit all-anchor semantic review is required')
        ss=a['source_sentences']
        if len(ss) not in (2,3) or ' '.join(ss)!=a['chunk_text']:
            raise ValueError('Expected 2–3 consecutive source sentences')
        if len(a['chunk_text'].split())>70:
            raise ValueError('Passage exceeds 70 words')
        for s in ss:
            if fingerprint(s) in sentences:raise ValueError('Repeated source sentence')
        labels={anchor['id']:int(i+1 in positives) for i,anchor in enumerate(anchors)}
        row=dict(a,chunk_id=passage_id(a['chunk_text']),working_labels=labels,draft_labels=dict(labels),
                 labels={anchor['id']:None for anchor in anchors},review_state='AI draft',
                 label_method=a['review_method'],anchor_version=version,split=None,
                 origins=[{'origin_key':'collected_'+a['candidate_id'],'origin_file':'collection/reviews.jsonl'}],
                 source_status='verified_exact_snapshot',source_verified=True,
                 review_note=a.get('note','Assistant semantic review against all 40 anchors. Human review pending.'))
        rows.append(row);keys.add(key);counts[a['source_id']]+=1
        sentences.update(fingerprint(s) for s in ss)
    return rows


def refresh(out=DEFAULT_OUT, csv_path=None, additions=None, xlsx_path=None):
    out=Path(out);document=json.loads((out/'anchors.json').read_text());rows=read_jsonl(out/'passages.jsonl')
    if xlsx_path:
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'workbook_labels.csv';p.write_text(read_xlsx_labels(xlsx_path,document['anchors']))
            rows=apply_csv(rows,p,document['anchors'])
    else:
        rows=apply_csv(rows,csv_path or out/'labeling.csv',document['anchors'])
    if additions:
        incoming=read_jsonl(additions)
        for record in incoming:validate_snapshot(record,out)
        rows=append_reviewed(rows,incoming,document['anchors'],document['anchor_version'])
    close=near_duplicates(rows)
    for r in rows:
        r['near_duplicate_ids']=[p['chunk_id_b'] if p['chunk_id_a']==r['chunk_id'] else p['chunk_id_a'] for p in close if r['chunk_id'] in (p['chunk_id_a'],p['chunk_id_b'])]
    # Keep a recovery copy before replacing editable data.
    backup=out/'backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');backup.mkdir(parents=True)
    for name in ('passages.jsonl','labeling.csv','summary.json'):
        shutil.copy2(out/name,backup/name)
    with (out/'duplicates.csv').open(encoding='utf-8-sig',newline='') as f:dups=list(csv.DictReader(f))
    with (out/'changed_target_labels.csv').open(encoding='utf-8-sig',newline='') as f:conflicts=list(csv.DictReader(f))
    prior=json.loads((out/'summary.json').read_text())
    return write_outputs(out,rows,document,dups,close,conflicts,target=prior['target_passages'],min_positive=prior['min_positive'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    g=p.add_mutually_exclusive_group();g.add_argument('--csv',type=Path);g.add_argument('--xlsx',type=Path)
    p.add_argument('--append',type=Path)
    a=p.parse_args();print(json.dumps(refresh(a.out,a.csv,a.append,a.xlsx),indent=2))

if __name__=='__main__':main()
