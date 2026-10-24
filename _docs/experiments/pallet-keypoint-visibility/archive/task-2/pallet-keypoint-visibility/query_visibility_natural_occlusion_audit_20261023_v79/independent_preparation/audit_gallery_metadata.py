"""Independent fixed-four selection, saved scores, native teacher and crop counts."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
from PIL import Image
from independent_grid_math import replay_masks
R=Path(__file__).resolve().parent;V=R.parent;P=V.parent;G=V/'natural_failure_gallery';OLD=P/'query_visibility_reference_local_support_20261023_v76';V77=P/'query_visibility_controlled_pairs_20261023_v77';V78=P/'query_visibility_dino_query_support_20261023_v78'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256']
policy=load(R/'INDEPENDENT_REPLAY_POLICY_BEFORE_OUTPUTS.json');assert policy['status'].startswith('FROZEN_BEFORE')
lock=load(G/'SELECTION_LOCK_BEFORE_RGB.json');assert sha(G/'SELECTION_LOCK_BEFORE_RGB.json')=='8072abb0456c155e3a9ba622045892a815e6ce869af996f4e92298627cf54ac0'
for b in lock['bindings']:check(b)
with np.load(OLD/'gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz',allow_pickle=False)as z:gt={k:z[k]for k in ['id','source','split','label8','native_GTref8','rule8','convex_self_proxy8_separate','center_M08','center_M48','rho_px']}
qrows=[json.loads(x)for x in (OLD/'public_query_emulation/INPUTS_RGB_ONLY.jsonl').read_text().splitlines()];qmap={r['id']:r for r in qrows};imap={x:i for i,x in enumerate(gt['id'])}
with (V77/'source_failure_audit/FIXED_COHORT_IDS.csv').open()as f:cohort=[x for x in csv.DictReader(f)if x['cohort']=='STRONG_AND_ERROR_LE3PX_AND_CENTER_CELL']
assert len(cohort)==535
scores={}
for name,path in [('V76',OLD/'mixed_fit/runs/MIXED_SUPPORT_V1/SELECTED_SOURCE_SCORES.npz'),('V77',V77/'runs/PAIRED_SUPPORT_V1/SELECTED_SOURCE_SCORES.npz'),('V78',V78/'runs/SOURCE_DINO_V1/SELECTED_SOURCE_SCORES.npz')]:
 with np.load(path,allow_pickle=False)as z:
  assert np.array_equal(z['id'][:5203],gt['id']);scores[name]=z['support_logit'][:5203];
  if name=='V76':primary=z['primary_loss_mask'][:5203]
  else:assert np.array_equal(primary,z['primary_loss_mask'][:5203])
pools={};totalcounts={0:0,1:0}
for r in cohort:
 i=imap[r['id']];k=int(r['native_k']);q=np.asarray(qmap[r['id']]['initial_points'])[k];ref=gt['native_GTref8'][i,k];L=qmap[r['id']]['roi_side'];y=int(gt['label8'][i,k]);pred=int(scores['V78'][i,k]>=0)
 assert gt['source'][i]=='YCB_PUBLIC_REAL'and gt['split'][i]=='val'and primary[i,k]and np.linalg.norm(q-ref)<=3 and np.max(abs((q-ref)/L))<=1/18 and str(gt['rule8'][i,k]).startswith('STRONG_3X3')
 h=hashlib.sha256((lock['selection_salt']+'\n'+r['id']+'\n'+str(k)).encode()).hexdigest();pools.setdefault((y,pred),[]).append((h,i,k));totalcounts[y]+=1
assert totalcounts=={0:76,1:459}
metadata=load(G/'GALLERY_METADATA.json');assert metadata['selection_lock']==bind(G/'SELECTION_LOCK_BEFORE_RGB.json')and metadata['completed']
man=load(OLD/'public_input_pack/PACK_MANIFEST.json');checks=[]
for c,m in zip(lock['cases'],metadata['cases']):
 y=c['GT_label'];pred=c['V78_prediction'];h,i,k=min(pools[y,pred]);ident=str(gt['id'][i]);qrow=qmap[ident]
 assert h==c['selection_SHA256']and ident==c['id']and k==c['native_k']and i==c['corpus_row']and c['cell_population']==len(pools[y,pred])
 for key in c:assert m[key]==c[key],key
 assert np.array_equal(c['q'],np.asarray(qrow['initial_points'])[k])and np.array_equal(c['r'],gt['native_GTref8'][i,k])and c['L']==qrow['roi_side']and c['box']==qrow['box']
 for name,zs in scores.items():
  z=float(zs[i,k]);s=c['scores'][name];assert s['logit']==z and s['p_visible']==float(1/(1+np.exp(-z)))and s['prediction']==('VISIBLE'if z>=0 else'COVERED')
 assert c['rule']==gt['rule8'][i,k]and c['self_visible_proxy_separate']==gt['convex_self_proxy8_separate'][i,k]
 check(c['source_RGB']);check(m['mask_pair']['M0']);check(m['mask_pair']['M4']);raw0=np.asarray(Image.open(m['mask_pair']['M0']['path']));raw4=np.asarray(Image.open(m['mask_pair']['M4']['path']));f=raw0>0;v=raw4>0;assert f.shape==v.shape and not(v&~f).any()
 x,yy=np.floor(np.asarray(c['r'])+.5).astype(int);assert f[yy-1:yy+2,x-1:x+2].all()
 assert (not v[yy-1:yy+2,x-1:x+2].any())if y==0 else v[yy-1:yy+2,x-1:x+2].all()
 pi=i-1003;chunk=next(x for x in man['chunks']if x['start']<=pi<x['end']);check(chunk['file'])
 with np.load(chunk['file']['path'],allow_pickle=False)as z:cached=z['crop_valid96'][pi-chunk['start'],k,0];assert np.array_equal(z['query8_xy'][pi-chunk['start'],k],c['q'])and float(z['L'][pi-chunk['start']])==c['L']
 counts=replay_masks(f,v,c['q'],c['L'],96,cached,'rgb96')['counts']['native'];published=m['local96_counts'];assert counts==dict(valid=published['native_label_valid'],V=published['visible'],C=published['nonvisible_expected'],O=published['outside_expected'])
 checks.append(dict(cell=c['cell'],id=ident,native_k=k,cell_population=len(pools[y,pred]),minSHA_exact=True,all3_saved_scores_exact=True,q_reference_L_box_teacher_exact=True,native3x3_teacher_replayed=True,scalar_RGB96_native_mask_counts=counts))
for c in metadata['cases']:
 for key in ['raw_local_RGB96','raw_target_RGB128','M0_display','M4_display','scene_overview','panel']:check(c[key])
check(metadata['contact_sheet'])
out=V/'independent_audits';out.mkdir(exist_ok=True);p=out/'GALLERY_SELECTION_AND_METADATA_AUDIT.json';assert not p.exists();p.write_text(json.dumps(dict(PASS=True,code=bind(Path(__file__)),replay_policy=bind(R/'INDEPENDENT_REPLAY_POLICY_BEFORE_OUTPUTS.json'),selection_lock=bind(G/'SELECTION_LOCK_BEFORE_RGB.json'),gallery=bind(G/'GALLERY_METADATA.json'),strict_cohort_N=535,GT_C76_V459=True,four_cells_minSHA_and_identity_scores_teacher_replayed=True,checks=checks,RGB_bodies_decoded=0,native_mask_bodies_decoded=8,source_cached_RGB_P3_decoded=0,new_semantic_GT_or_label_correction=False,selection_outcome_conditioned_not_prevalence=True,source_model_forward_fit_current_PnP=0),indent=2)+'\n');print(json.dumps(dict(PASS=True,report=bind(p))))
