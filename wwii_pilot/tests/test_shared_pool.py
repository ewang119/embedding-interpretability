import copy
import csv
import tempfile
import unittest
from pathlib import Path
from wwii_pilot.build_shared_pool import fingerprint,passage_id,merge_records,apply_judgments,coverage,group_source
from wwii_pilot.refresh_shared_pool import parse_label,append_reviewed,apply_csv

A=[{'id':'a','code':'A01','name':'One','definition_status':'user_supplied'}, {'id':'b','code':'A02','name':'Two','definition_status':'provisional_inferred'}]
class SharedPoolTests(unittest.TestCase):
 def test_identity_normalizes_typography_but_not_words(self):
  self.assertEqual(passage_id('“War”—production.'),passage_id('war production'))
  self.assertNotEqual(passage_id('War production'),passage_id('War rationing'))
 def test_missing_review_is_not_negative(self):
  rows,_=merge_records([dict(chunk_text='Test passage.',origin_key='x',original_anchor_number=1,original_target_label=1)],[])
  apply_judgments(rows,A,{})
  self.assertEqual(rows[0]['working_labels'],{'a':None,'b':None})
  self.assertEqual(coverage(rows,A)[0]['unreviewed'],1)
 def test_duplicate_conflicting_judgments_do_not_become_labels(self):
  rows,dups=merge_records([dict(chunk_text=t,origin_key=k,original_anchor_number=1,original_target_label=1) for t,k in [('Same text.','x'),('same text!','y')]],[])
  apply_judgments(rows,A,{'x':{'positive_anchor_numbers':[1]},'y':{'positive_anchor_numbers':[2]}})
  self.assertEqual(len(rows),1);self.assertEqual(len(dups),1)
  self.assertEqual(rows[0]['review_state'],'Conflict');self.assertIsNone(rows[0]['working_labels']['a'])
 def test_complete_semantic_review_and_binary_types(self):
  rows,_=merge_records([dict(chunk_text='A passage.',origin_key='x',original_anchor_number=1,original_target_label=1)],[])
  apply_judgments(rows,A,{'x':{'positive_anchor_numbers':[2]}})
  self.assertEqual(rows[0]['working_labels'],{'a':0,'b':1});self.assertEqual(rows[0]['labels'],{'a':None,'b':None})
  with self.assertRaises(ValueError):apply_judgments(rows,A,{'x':{'positive_anchor_numbers':[True]}})
 def test_ambiguous_source_is_not_invented(self):
  rows,_=merge_records([dict(chunk_text='A passage.',origin_key='x')],[dict(chunk_text='A passage.',source_id=s,source_url='https://example.org/'+s) for s in ('one','two')])
  self.assertIsNone(rows[0].get('source_url'));self.assertEqual(rows[0]['source_match'],'ambiguous_candidate_sources')
 def test_whole_books_share_source_group(self):
  urls=['https://x/USA-E-Ardennes/chapter1.html','https://x/USA-E-Ardennes/chapter2.html']
  self.assertEqual(*[group_source({'source_url':u}) for u in urls])
 def test_blank_is_not_zero(self):
  self.assertIsNone(parse_label(''));self.assertEqual(parse_label('0'),0)
  for bad in ('false','0.0','2'):
   with self.assertRaises(ValueError):parse_label(bad)
 def test_new_sources_capped_and_sentences_cannot_repeat(self):
  a=dict(chunk_text='The workers built aircraft. The factory operated continuously.',source_sentences=['The workers built aircraft.','The factory operated continuously.'],candidate_id='x',source_id='s',source_url='https://x',positive_anchor_numbers=[1],review_method='assistant_semantic_review_all_anchors')
  rows=append_reviewed([], [a], A,'v')
  self.assertEqual(rows[0]['working_labels'],{'a':1,'b':0})
  b=dict(a,chunk_text='The workers built aircraft. A second sentence is here.',source_sentences=['The workers built aircraft.','A second sentence is here.'])
  with self.assertRaises(ValueError):append_reviewed(rows,[b],A,'v')
  with self.assertRaises(ValueError):append_reviewed([{'chunk_text':str(i),'source_id':'s'} for i in range(3)],[a],A,'v')

class SharedPoolEditingTests(unittest.TestCase):
 def write_edits(self,path,labels=('0',''),state='AI draft'):
  fields=['passage_id','passage_text','review_state','source_id','source_url','source_verified','a','b']
  with path.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(dict(zip(fields,['p','An unchanged passage.',state,'s','https://example.org','no',*labels])))
 def row(self):
  return dict(chunk_id='p',chunk_text='An unchanged passage.',working_labels={'a':1,'b':0},review_state='AI draft')
 def test_csv_blank_and_negative_remain_distinct(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'labels.csv';self.write_edits(p)
   r=apply_csv([self.row()],p,A)[0]
   self.assertEqual(r['working_labels'],{'a':0,'b':None});self.assertEqual(r['labels'],{'a':None,'b':None})
 def test_incomplete_row_cannot_be_approved(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'labels.csv';self.write_edits(p,state='Reviewed')
   with self.assertRaisesRegex(ValueError,'blank labels'):apply_csv([self.row()],p,A)
 def test_approved_row_and_text_identity(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'labels.csv';self.write_edits(p,('1','0'),'Reviewed')
   r=apply_csv([self.row()],p,A)[0];self.assertEqual(r['labels'],{'a':1,'b':0})
   p.write_text(p.read_text().replace('An unchanged passage.','An edited passage.'))
   with self.assertRaisesRegex(ValueError,'Text changed'):apply_csv([self.row()],p,A)
 def test_source_snapshot_must_match_and_stay_inside_pool(self):
  import json
  from wwii_pilot.refresh_shared_pool import validate_snapshot
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'source.json').write_text(json.dumps({'sha256':'abc','paragraphs':['One sentence. Another sentence.']}))
   record=dict(source_snapshot='source.json',source_html_sha256='abc',paragraph_index=0,char_start=0,char_end=31,chunk_text='One sentence. Another sentence.',source_sentences=['One sentence.','Another sentence.'])
   self.assertTrue(validate_snapshot(record,root))
   with self.assertRaisesRegex(ValueError,'exactly match'):validate_snapshot(dict(record,chunk_text='One invented sentence.'),root)
   with self.assertRaisesRegex(ValueError,'inside'):validate_snapshot(dict(record,source_snapshot='../source.json'),root)

if __name__=='__main__':unittest.main()
