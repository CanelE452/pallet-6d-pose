"""Focused artifact, support and arithmetic checks. Never loads a model or raw cache."""
import copy,csv,json,math,re,statistics,unittest
import numpy as np
from PIL import Image
import report as R

class CloseoutTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.d,cls.pin,cls.edges=R.load();cls.s=R.read(R.DOC/'SUMMARY.json');cls.m=R.read(R.DOC/'REPORT_MANIFEST.json')
 def test_01_source_and_output_bindings(self):
  for b in self.pin['inputs'].values():R.verify_binding(b)
  for b in self.m['outputs'].values():R.verify_binding(b)
  for b in self.m['codes']:R.verify_binding(b)
  self.assertGreaterEqual(len(self.edges),20)
 def test_02_reject_tampered_hash_and_size(self):
  b=copy.deepcopy(self.pin['inputs']['yolo']);b['sha256']='0'*64
  with self.assertRaises(AssertionError):R.verify_binding(b)
  b=copy.deepcopy(self.pin['inputs']['yolo']);b['bytes']+=1
  with self.assertRaises(AssertionError):R.verify_binding(b)
  b=copy.deepcopy(self.pin['inputs']['yolo']);del b['bytes'];R.verify_binding(b)
 def test_03_every_accuracy_number_and_seed_mean(self):
  for row in self.s['accuracy_rows']:
   key={'YOLO':'yolo','DOPE':'dope','ResNet18':'resnet'}[row['backbone']]
   methods=self.d[key]['methods'];arm=row['method']
   names=[arm] if row['seed_count']==1 else [f'{arm}_S{i}' if key=='resnet' else f'{arm}{i}' for i in (1,2,3)]
   raw=[methods[n] for n in names]
   for field in ['median_px','p90_px','matched_frames','supervised_points']:
    out='conditional_landmarks' if field=='supervised_points' else field
    self.assertAlmostEqual(row[out],float(np.mean([x[field] for x in raw])),places=12)
   for field in ['translation_median_cm','rotation_median_deg','coverage']:
    out='pose_coverage' if field=='coverage' else field
    self.assertAlmostEqual(row[out],float(np.mean([x['pose'][field] for x in raw])),places=12)
   self.assertAlmostEqual(row['all_gt_pck10'],float(np.mean([x['ALL_GT_PCK']['10'] for x in raw])),places=14)
  self.assertEqual([(r['matched_frames'],r['conditional_landmarks']) for r in self.s['accuracy_rows'] if r['seed_count']==1],[(311,2756),(190,1710),(299,2644)])
 def test_04_full_runtime_rows_and_failures_recompute(self):
  rows=self.d['runtime']['measurements'];self.assertEqual(len(rows),1170)
  for out in self.s['runtime_rows']:
   selected=[x for x in rows if x['arm']==out['arm']]
   self.assertEqual(len(selected),130)
   self.assertEqual(len({x['frame_id'] for x in selected}),26)
   self.assertEqual(len({(x['repeat'],x['frame_id']) for x in selected}),130)
   for field,metric in [('two_d_median_ms','two_d_ms'),('full_median_ms','full_ms'),('pnp_median_ms','pnp_ms')]:
    self.assertAlmostEqual(out[field],statistics.median(x[metric] for x in selected),places=12)
   self.assertAlmostEqual(out['full_p90_ms'],float(np.percentile([x['full_ms'] for x in selected],90)),places=12)
   counts={}
   for x in selected:
    counts[x['pose_status']]=counts.get(x['pose_status'],0)+1
    self.assertAlmostEqual(x['full_ms'],x['two_d_ms']+x['pnp_ms'],places=9)
   self.assertEqual(counts,out['pose_status_counts'])
 def test_05_overhead_definition_and_main_arm(self):
  by={x['arm']:x for x in self.s['runtime_rows']}
  for row in self.s['runtime_overhead']:
   a,b=by[row['baseline']],by[row['correction']]
   self.assertEqual(row['delta_full_median_ms'],b['full_median_ms']-a['full_median_ms'])
   raw=self.d['runtime']['measurements']
   base={(r['repeat'],r['frame_id']):r for r in raw if r['arm']==row['baseline']}
   corrected={(r['repeat'],r['frame_id']):r for r in raw if r['arm']==row['correction']}
   self.assertEqual(base.keys(),corrected.keys());self.assertEqual(len(base),130)
   for source,out in [('two_d_ms','paired_added_two_d_median_ms'),('full_ms','paired_added_full_median_ms')]:
    expected=statistics.median(corrected[k][source]-base[k][source] for k in base)
    self.assertAlmostEqual(row[out],expected,places=12)
   self.assertNotEqual(row['paired_added_full_median_ms'],row['delta_full_median_ms'])
   self.assertEqual(row['paired_observations'],130)
  self.assertEqual(self.s['runtime_overhead'][2]['correction'],'RESNET_P0_S1')
  self.assertEqual([r['method'] for r in self.s['accuracy_rows'] if r['main_comparison']],['R0','P','DOPE','P','FULL','P0'])
 def test_06_direct_endpoints_and_physical_frame_not_mixed(self):
  for row in self.s['direct_corner8_rows']:
   src=self.d['direct_'+row['arm']]['results'][row['population']]['summary']
   self.assertEqual(row['median_corner8_px'],src['matched_pooled_corner8_median_px'])
   self.assertEqual(row['matched_frames'],src['matched'])
   self.assertEqual(row['declared_corner_landmarks'],src['corners'])
  canonical=next(r for r in self.s['accuracy_rows'] if r['method']=='FULL')
  direct=next(r for r in self.s['direct_corner8_rows'] if r['arm']=='FULL' and r['population']=='DEV319')
  self.assertNotEqual(canonical['median_px'],direct['median_corner8_px'])
  p=next(x for x in self.s['direct_pose_rows'] if x['arm']=='FULL')
  self.assertNotEqual(canonical['rotation_median_deg'],p['rotation_median_deg'])
 def test_07_causal_claim_preserves_intervals(self):
  selected=[x for x in self.s['paired_rows'] if x['contrast']=='P5_minus_P5_CONSTANT']
  self.assertEqual(len(selected),4)
  for x in selected:
   ref=next(r for r in self.d['resnet_tables']['paired_session_rows'] if r['contrast']==x['contrast'] and r['metric']==x['metric'])
   for k in ['delta','ci95_low','ci95_high']:self.assertEqual(x[k],ref[k])
   if x['metric']!='ALL_GT_PCK10':self.assertLess(x['ci95_low'],0);self.assertGreater(x['ci95_high'],0)
  self.assertFalse(self.s['stable_joint_TR_established']);self.assertFalse(self.s['independent_test_present'])
 def test_08_every_csv_cell_matches_full_precision_summary(self):
  for file,key in [('ACCURACY_SUMMARY.csv','accuracy_rows'),('RUNTIME_SUMMARY.csv','runtime_rows'),('RUNTIME_OVERHEAD.csv','runtime_overhead'),('PAIRED_SUMMARY.csv','paired_rows'),('DIRECT_DIMENSION_SUMMARY.csv','direct_corner8_rows'),('DIRECT_POSE_SUMMARY.csv','direct_pose_rows')]:
   with (R.DOC/file).open() as f:rows=list(csv.DictReader(f))
   self.assertEqual(len(rows),len(self.s[key]))
   for actual,expected in zip(rows,self.s[key]):
    for k,v in expected.items():
     serial=json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(list,dict)) else str(v)
     self.assertEqual(actual[k],serial)
 def test_09_markdown_numeric_tables_and_links(self):
  text=(R.DOC/'REPORT_KO.md').read_text()
  for row in self.s['accuracy_rows']:
   expected=f"| {row['backbone']} | {row['method']} | {row['median_px']:.3f} | {row['p90_px']:.3f} | {100*row['all_gt_pck10']:.2f} | {row['matched_frames']:.0f}/319 | {row['translation_median_cm']:.3f} | {row['rotation_median_deg']:.3f} | {100*row['pose_coverage']:.2f} |"
   self.assertIn(expected,text)
  for r in self.s['paired_rows']:
   self.assertIn(f"{r['delta']:+.6f} [{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",text)
  self.assertGreater(len(R.validate_links(text)),20)
  for row in self.s['runtime_overhead']:
   self.assertIn(f"{row['paired_added_two_d_median_ms']:.3f}",text)
   self.assertIn(f"{row['paired_added_full_median_ms']:.3f}",text)
  self.assertIn('130개 paired 차이의 median',text)
  self.assertIn('pallet_resnet18_dim_refiner_report_20261002_v3/REPORT_KO.md',text)
  for phrase in ['R_physical','R_cf','P5−P5_CONSTANT','독립 TEST','0을 포함','corner8']:
   self.assertIn(phrase,text)
 def test_10_figures_and_existing_rgb(self):
  for name in ['within_backbone_accuracy','runtime_overhead']:
   with Image.open(R.DOC/'figures'/f'{name}.png') as im:self.assertGreater(im.width,1500);self.assertGreater(im.height,600);im.verify()
   self.assertTrue((R.DOC/'figures'/f'{name}.pdf').read_bytes().startswith(b'%PDF'))
  for alias in [k for k in R.SOURCES if k.startswith('rgb_')]:
   with Image.open(R.EXP/R.SOURCES[alias]) as im:self.assertGreater(im.width,100);im.verify()
  self.assertEqual(self.m['figure_values']['runtime_overhead'],self.s['runtime_overhead'])
 def test_11_only_public_bounded_inputs_no_training(self):
  for b in self.pin['inputs'].values():
   self.assertTrue(b['path'].startswith('_docs/experiments/'))
   self.assertNotIn('.pt',b['path']);self.assertLess(b['bytes'],20_000_000)
  for key in ['new_fits','new_image_forwards','new_pose_solves','new_bootstrap_runs']:self.assertEqual(self.m[key],0)
  self.assertEqual(self.m['checkpoints_read_or_copied'],0);self.assertFalse(self.m['manuscript_modified'])

if __name__=='__main__':
 R.write('TEST_RESULTS.json',dict(complete=False,status='RUNNING',scope='same-producer focused arithmetic/provenance checks'))
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(CloseoutTests)
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 R.write('TEST_RESULTS.json',dict(complete=True,PASS=result.wasSuccessful(),tests_run=result.testsRun,
  failures=[(str(t),e) for t,e in result.failures],errors=[(str(t),e) for t,e in result.errors],
  scope='Existing source byte bindings, seed means, all1170runtime calls, supports, direct/canonical split, causal intervals, CSV cells, Markdown, images and relative links. Not independent model/GT replication.',
  report=R.bind(R.DOC/'REPORT_KO.md'),manifest=R.bind(R.DOC/'REPORT_MANIFEST.json'),test_code=R.bind(__file__),new_gpu_calls=0))
 raise SystemExit(0 if result.wasSuccessful() else 1)
