"""Import WWII collections, retain provenance, and monitor multi-label review.

The source topic bucket is evidence about one label, never implicit negatives for other anchors.
All numeric AI labels below come from an explicit passage-level judgment file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'wwii_pilot/data'
ANCHOR_IDS = [
    'eastern_front_attrition_logistics', 'siege_encirclement_warfare',
    'pacific_amphibious_campaigns', 'atlantic_convoy_warfare',
    'strategic_bombing_campaigns', 'normandy_landings', 'appeasement_germany',
    'munich_agreement', 'nazi_soviet_pact_fallout', 'us_neutrality_intervention',
    'japanese_expansionism', 'allied_coalition_diplomacy', 'postwar_settlement_planning',
    'persecution_jews_before_war', 'holocaust_extermination_camps',
    'occupation_civilian_reprisals', 'collaboration_resistance', 'pow_treatment',
    'forced_labor', 'war_crimes_prosecution', 'mobilization_industry',
    'lend_lease_supply', 'blockade_resource_scarcity', 'axis_exploitation_occupied_economies',
    'oil_fuel_constraints', 'aircraft_shipbuilding_output', 'churchill_rhetoric_public_morale',
    'hitler_interference_military_command', 'allied_command_coordination',
    'stalin_purges_red_army', 'codebreaking_ultra', 'espionage_deception',
    'atomic_bomb_development', 'radar_air_defense', 'rocketry_v_weapons',
    'rationing_civilian_austerity', 'women_wartime_labor', 'civilian_bombing_sheltering',
    'propaganda_information_control', 'japanese_american_internment',
]
DRAFT_DEFINITIONS = [
    'Substantially about WWII cryptanalysis, intercepted or decrypted enemy communications, Enigma, Ultra, Bletchley Park, or related Allied/Axis codebreaking, including operational use of decrypted intelligence. Espionage or deception alone does not qualify.',
    'Substantially about WWII human espionage, counterintelligence, double agents, or deliberate military deception such as Fortitude, Mincemeat, dummy forces, and misleading an opponent. Codebreaking alone does not qualify without espionage or deception.',
    'Substantially about developing, funding, designing, manufacturing, or testing nuclear weapons for WWII, including the Manhattan Project, fissile-material plants, Los Alamos, and Trinity. Bombing casualties or surrender debates without a development focus do not qualify.',
    'Substantially about wartime radar detection, early warning, tracking, or organized air defense, including the integration of radar, observers, fighter interception, and anti-aircraft defenses. Offensive bombing alone does not qualify.',
    'Substantially about wartime development, manufacture, deployment, or effects of the V-1, V-2, and other rockets or pilotless V-weapons. Generic aircraft manufacturing or artillery does not qualify.',
    'Substantially about civilian rationing, consumption restrictions, shortages as experienced by households, ration books, substitute goods, or home-front austerity during WWII. Industrial scarcity, blockades, and military fuel logistics alone do not qualify.',
    'Substantially about women entering or performing wartime work in factories, shipyards, farms, offices, or auxiliary military services, including working conditions and changing labor roles. Merely mentioning women among civilians or victims does not qualify.',
    'Substantially about civilians experiencing WWII air raids, casualties, damaged homes, sheltering, evacuation from bombing, or psychological and social effects. Operational bombing plans without substantial civilian experience or consequences do not qualify.',
    'Substantially about wartime public persuasion, propaganda, morale messaging, censorship, manipulation of news, or information control through speeches, broadcasts, films, newspapers, posters, or leaflets. Mentioning a leader or private intelligence alone does not qualify.',
    'Substantially about WWII removal, incarceration, camp conditions, legal challenges, or directly related consequences for people of Japanese ancestry in the United States. Japanese occupation camps and Allied military POW camps are different topics.',
    'Substantially about civilian evacuation, refugees, forced displacement, flight from persecution or fighting, immigration restrictions, or resettlement associated with the WWII era. Military withdrawals such as Dunkirk and moving factories alone do not qualify.',
]


def fingerprint(text):
    """Ignore typography/spacing for identity, while retaining the original text."""
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def passage_id(text):
    return 'wwii_' + hashlib.sha256(fingerprint(text).encode()).hexdigest()[:16]


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write_csv(path, fields, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_anchors(path):
    lines = [s[2:].strip() for s in Path(path).read_text().splitlines()
             if s.startswith('- ') and not s.startswith('- →')]
    # Exclude the optional 41st anchor from the working set.
    lines = [s for s in lines if s != 'Evacuation and refugees']
    if len(lines) != len(ANCHOR_IDS):
        raise ValueError(f'Expected {len(ANCHOR_IDS)} anchor names, found {len(lines)}')
    anchors = []
    for i, (line, aid) in enumerate(zip(lines, ANCHOR_IDS)):
        name, sep, definition = line.partition(':')
        name = re.sub(r'\s*\(K\)$', '', name).strip()
        if not sep:
            definition = DRAFT_DEFINITIONS[i - 30]
        anchors.append(dict(id=aid, code=f'A{i+1:02}', name=name,
                            definition=definition.strip(),
                            definition_status='user_supplied' if sep else 'provisional_inferred',
                            source_note='Arrow note under codebreaking retained as scope guidance, not a separate anchor.' if i == 30 else ''))
    return dict(schema_version=1, anchor_version='wwii-full-draft-2026-11-17',
                omitted_anchor_names=['Evacuation and refugees'],
                note='40 anchors; Evacuation and refugees is excluded. Last 10 definitions are provisional.',
                anchors=anchors)


def parse_batch(path, anchors):
    lookup = {fingerprint(a['name']): i+1 for i,a in enumerate(anchors)}
    current = None
    label = 1
    counts = Counter()
    records = []
    for line_no, raw in enumerate(Path(path).read_text().splitlines(), 1):
        text = raw.strip()
        if not text:
            continue
        heading = re.match(r'^\d+\.\s+(.+)$', text)
        if heading:
            key = fingerprint(heading.group(1))
            if key not in lookup:
                raise ValueError(f'Unknown anchor heading on line {line_no}: {text}')
            current, label = lookup[key], 1
            continue
        if text.lower() == 'hard negatives':
            label = 0
            continue
        if not current:
            continue
        if text in {'Campaigns','Politics & Diplomacy','Human Rights','Economics','Economy','Leadership'}:
            continue
        # Plain section headings never become fake passages.
        if len(text.split()) < 5 and not re.search(r'[.!?]$', text):
            continue
        counts[current] += 1
        ordinal = counts[current]
        records.append(dict(chunk_text=text,
                            origin_key=f'a{current:02}_{ordinal:02}',
                            original_anchor_number=current, original_target_label=label,
                            origin_file=Path(path).name, origin_line=line_no,
                            source_id=None, source_url=None, source_title=None))
    for anchor, n in counts.items():
        if n != 15:
            raise ValueError(f'Anchor {anchor} has {n} excerpts, expected 15')
        labels = [r['original_target_label'] for r in records if r['original_anchor_number']==anchor]
        if labels != [1]*12+[0]*3:
            raise ValueError(f'Unexpected positive/negative boundary for anchor {anchor}')
    return records


def candidate_sources(directory):
    """An explicit manifest avoids re-importing superseded versions."""
    names = [
        'mobilization_12_positive_3_negative_2sent',
        'lend_lease_12_positive_3_negative_2to3sent',
        'blockade_resource_scarcity_12_positive_3_negative_2to3sent',
        'axis_exploitation_occupied_economies_12_positive_3_negative_2to3sent',
        'oil_fuel_constraints_operations_12_positive_3_negative_2to3sent',
        'aircraft_shipbuilding_output_12_positive_3_negative_2to3sent',
        'churchill_rhetoric_public_morale_12_positive_3_negative_2to3sent',
        'hitler_interference_military_command_12_positive_3_negative_2to3sent',
        'allied_command_coordination_12_positive_3_negative_2to3sent',
        'stalin_purges_red_army_12_positive_3_negative_2to3sent',
    ]
    result = []
    for number, name in enumerate(names, 21):
        p = Path(directory)/(name+'.jsonl')
        for i,r in enumerate(read_jsonl(p),1):
            result.append(dict(r, candidate_file=p.name, candidate_ordinal=i,
                               candidate_anchor_number=number))
    return result


def group_source(row):
    """Group chapters of an underlying book together, not just individual URLs."""
    if row.get('source_work_id'):
        return row['source_work_id']
    url = row.get('source_url') or ''
    for pattern in [r'/(USA-[^/]+)/', r'/(UK-[^/]+)/', r'/(AAF)/(I|II|III|IV|V|VI|VII)/']:
        m = re.search(pattern, url)
        if m:
            return 'book_' + '_'.join(m.groups())
    m = re.search(r'/historicaldocuments/(frus[^/]+)/', url)
    if m:
        return 'book_' + m.group(1)
    if '/USSBS/IJO/' in url:
        return 'book_USSBS_Interrogations_Japanese_Officials'
    if row.get('source_id'):
        return row['source_id']
    return None


def merge_records(records, provenance):
    lookup = defaultdict(list)
    for r in provenance:
        lookup[fingerprint(r['chunk_text'])].append(r)
    unique = {}
    duplicates = []
    for r in records:
        key = fingerprint(r['chunk_text'])
        matches = lookup.get(key, [])
        source_ids = {group_source(m) for m in matches if group_source(m)}
        if len(source_ids) == 1:
            m = matches[0]
            r.update(source_id=next(iter(source_ids)), source_url=m.get('source_url'),
                     source_title=m.get('source_title'),
                     source_match='normalized_exact_candidate_match',
                     candidate_file=m.get('candidate_file'))
        elif len(source_ids) > 1:
            r['source_match'] = 'ambiguous_candidate_sources'
        origin = {k:r[k] for k in ['origin_key','origin_file','origin_line',
                                  'original_anchor_number','original_target_label'] if k in r}
        if key in unique:
            kept=unique[key]
            kept['origins'].append(origin)
            if not kept.get('source_id') and r.get('source_id'):
                for field in ['source_id','source_url','source_title','source_match','candidate_file']:
                    kept[field] = r.get(field)
            duplicates.append(dict(kept_id=kept['chunk_id'], removed_origin=r['origin_key'],
                                   reason='same text after typography/case/whitespace normalization'))
            continue
        row=dict(r, chunk_id=passage_id(r['chunk_text']), origins=[origin],
                 source_status='candidate_provenance' if r.get('source_url') else 'missing_original_source',
                 source_verified=False, review_state='AI draft', split=None)
        unique[key]=row
    return list(unique.values()), duplicates


def apply_judgments(rows, anchors, judgments):
    valid=set(range(1,len(anchors)+1))
    conflicts=[]
    for row in rows:
        decisions=[]
        for origin in row['origins']:
            decision=judgments.get(origin['origin_key'])
            if decision is not None:
                positives=decision['positive_anchor_numbers']
                if any(type(n) is not int for n in positives) or len(set(positives))!=len(positives) or not set(positives)<=valid:
                    raise ValueError(f'Invalid anchors in {origin["origin_key"]}')
                decisions.append((origin,decision))
        if not decisions:
            row['working_labels']={a['id']:None for a in anchors}
            row['review_state']='Unreviewed'
            row['review_note']='No passage-level semantic review has been supplied.'
        elif len({tuple(sorted(d['positive_anchor_numbers'])) for _,d in decisions})>1:
            row['working_labels']={a['id']:None for a in anchors}
            row['review_state']='Conflict'
            row['review_note']='Duplicate text has conflicting draft reviews; reconcile before labeling.'
        else:
            positive=set(decisions[0][1]['positive_anchor_numbers'])
            row['working_labels']={a['id']:int(i+1 in positive) for i,a in enumerate(anchors)}
            row['review_note']=decisions[0][1].get('note','Assistant semantic review against all supplied definitions; numeric labels remain AI drafts.')
            row['label_method']='assistant_semantic_review_all_anchors'
        row['draft_labels']=dict(row['working_labels'])
        row['labels']={a['id']:None for a in anchors}
        for origin in row['origins']:
            aid=anchors[origin['original_anchor_number']-1]['id']
            current=row['working_labels'][aid]
            if current is not None and current!=origin['original_target_label']:
                conflicts.append(dict(chunk_id=row['chunk_id'],origin_key=origin['origin_key'],
                                      anchor_id=aid,original_target_label=origin['original_target_label'],
                                      proposed_label=current,chunk_text=row['chunk_text']))
    return conflicts


def near_duplicates(rows, threshold=.88):
    results=[]
    words=[set(fingerprint(r['chunk_text']).split()) for r in rows]
    for i,a in enumerate(rows):
        for j in range(i):
            inter=len(words[i]&words[j]); union=len(words[i]|words[j])
            if union and inter/union >= .65:
                ratio=SequenceMatcher(None,fingerprint(a['chunk_text']),fingerprint(rows[j]['chunk_text']),autojunk=False).ratio()
                if ratio>=threshold:
                    results.append(dict(chunk_id_a=rows[j]['chunk_id'],chunk_id_b=a['chunk_id'],
                                        similarity=round(ratio,4),action='review together; keep in same split if both retained'))
    return results


def coverage(rows, anchors, min_positive=75):
    result=[]
    for a in anchors:
        values=[r['working_labels'].get(a['id']) for r in rows]
        pos=sum(type(v) is int and v==1 for v in values)
        neg=sum(type(v) is int and v==0 for v in values)
        sources={r['source_id'] for r in rows if r['working_labels'].get(a['id'])==1 and r.get('source_id')}
        known=sum(r['working_labels'].get(a['id'])==1 and bool(r.get('source_url')) for r in rows)
        verified=sum(r['working_labels'].get(a['id'])==1 and r.get('source_verified',False) for r in rows)
        verified_sources={r['source_id'] for r in rows if r['working_labels'].get(a['id'])==1 and r.get('source_verified',False)}
        reviewed=sum(r['review_state']=='Reviewed' and r['working_labels'].get(a['id'])==1 for r in rows)
        result.append(dict(code=a['code'],anchor_id=a['id'],name=a['name'],
                           definition_status=a['definition_status'],draft_positive=pos,draft_negative=neg,
                           unreviewed=len(rows)-pos-neg,known_source_positive=known,
                           distinct_known_positive_sources=len(sources),verified_source_positive=verified,
                           distinct_verified_positive_sources=len(verified_sources),human_reviewed_positive=reviewed,
                           target_positive=min_positive,positive_shortfall=max(0,min_positive-pos)))
    return result


def write_outputs(out, rows, anchor_document, duplicates, close, conflicts, target=1000, min_positive=75):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    anchors=anchor_document['anchors']; aids=[a['id'] for a in anchors]
    (out/'anchors.json').write_text(json.dumps(anchor_document,ensure_ascii=False,indent=2)+'\n')
    (out/'passages.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    fields=['passage_id','passage_text','review_state','source_id','source_url','source_verified',*aids]
    labelrows=[dict(passage_id=r['chunk_id'],passage_text=r['chunk_text'],review_state=r['review_state'],
                    source_id=r.get('source_id') or '',source_url=r.get('source_url') or '',
                    source_verified='yes' if r.get('source_verified') else 'no',**r['working_labels']) for r in rows]
    write_csv(out/'labeling.csv',fields,labelrows)
    counts=coverage(rows,anchors,min_positive)
    write_csv(out/'coverage.csv',list(counts[0]),counts)
    write_csv(out/'collection_priorities.csv',list(counts[0]),sorted(counts,key=lambda r:(-r['positive_shortfall'],r['known_source_positive'],r['code'])))
    write_csv(out/'duplicates.csv',['kept_id','removed_origin','reason'],duplicates)
    write_csv(out/'near_duplicates.csv',['chunk_id_a','chunk_id_b','similarity','action'],close)
    write_csv(out/'changed_target_labels.csv',['chunk_id','origin_key','anchor_id','original_target_label','proposed_label','chunk_text'],conflicts)
    sources=[dict(passage_id=r['chunk_id'],source_id=r.get('source_id') or '',source_url=r.get('source_url') or '',
                  source_title=r.get('source_title') or '',source_status=r['source_status'],source_verified=r.get('source_verified',False),
                  origins='; '.join(o['origin_key'] for o in r['origins']),review_note=r['review_note']) for r in rows]
    write_csv(out/'sources.csv',list(sources[0]),sources)
    summary=dict(passages=len(rows),target_passages=target,remaining_passages=max(0,target-len(rows)),
                 anchors=len(anchors),provisional_anchor_definitions=sum(a['definition_status']=='provisional_inferred' for a in anchors),
                 draft_decisions=sum(v is not None for r in rows for v in r['working_labels'].values()),
                 human_reviewed_passages=sum(r['review_state']=='Reviewed' for r in rows),
                 known_source_passages=sum(bool(r.get('source_url')) for r in rows),
                 source_verified_passages=sum(bool(r.get('source_verified')) for r in rows),
                 missing_original_source=sum(not r.get('source_url') for r in rows),
                 duplicates_removed=len(duplicates),near_duplicate_pairs=len(close),changed_target_labels=len(conflicts),
                 min_positive=min_positive,split_status='unassigned: original source groups and reviewed labels must be completed before training export',
                 train_ready=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # One intermediate feeds the workbook; no independent copies of label values.
    (out/'workbook_data.json').write_text(json.dumps(dict(summary=summary,anchors=anchors,rows=labelrows,sources=sources,coverage=counts),ensure_ascii=False))
    return summary


def build(batch, anchor_notes, candidates, judgments, out, force=False):
    out=Path(out)
    if (out/'labeling.csv').exists() and not force:
        raise ValueError('labeling.csv already exists. Use refresh to preserve edits; --force explicitly replaces the working draft.')
    anchor_document=load_anchors(anchor_notes);anchors=anchor_document['anchors']
    records=parse_batch(batch,anchors);previous=candidate_sources(candidates)
    for r in previous:
        if r['candidate_anchor_number']>=27:
            records.append(dict(chunk_text=r['chunk_text'],origin_key=f'a{r["candidate_anchor_number"]:02}_{r["candidate_ordinal"]:02}',
                                original_anchor_number=r['candidate_anchor_number'],original_target_label=r['proposed_label'],
                                origin_file=r['candidate_file'],source_id=group_source(r),source_url=r['source_url'],source_title=r.get('source_title')))
    rows,duplicates=merge_records(records,previous)
    decisions=json.loads(Path(judgments).read_text()) if judgments else {}
    conflicts=apply_judgments(rows,anchors,decisions)
    close=near_duplicates(rows)
    for r in rows:
        r['near_duplicate_ids']=[p['chunk_id_b'] if p['chunk_id_a']==r['chunk_id'] else p['chunk_id_a'] for p in close if r['chunk_id'] in (p['chunk_id_a'],p['chunk_id_b'])]
        r['anchor_version']=anchor_document['anchor_version']
    summary=write_outputs(out,rows,anchor_document,duplicates,close,conflicts)
    inputs=out/'inputs'; inputs.mkdir(exist_ok=True)
    (inputs/'excerpts.txt').write_text(Path(batch).read_text())
    (inputs/'anchor_notes.txt').write_text(Path(anchor_notes).read_text())
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--anchor-notes',type=Path,required=True)
    parser.add_argument('--candidates',type=Path,default=ROOT/'wwii_pilot/inputs/candidate_sources')
    parser.add_argument('--judgments',type=Path)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    print(json.dumps(build(args.batch,args.anchor_notes,args.candidates,args.judgments,args.out,args.force),indent=2))


if __name__=='__main__':
    main()
