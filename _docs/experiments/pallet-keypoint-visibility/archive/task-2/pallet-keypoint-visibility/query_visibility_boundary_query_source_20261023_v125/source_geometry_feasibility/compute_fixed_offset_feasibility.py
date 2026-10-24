"""Read-only PUBLIC composition alpha feasibility. Outputs counts, never query/teacher manifests."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,math
import numpy as np
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');C=P/'query_visibility_flat_pallet_cue_source_20261023_v122';O=Path(__file__).resolve().parent
AMPS=(3,6,9,12,18,24) # axis amplitude, not Euclidean disk radius; geometry-only comparison.
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def main():
 output=O/'READ_ONLY_FEASIBILITY.json';assert not output.exists()
 ip=C/'actual_corpus/OCCURRENCES4896_INPUT_ONLY.jsonl';gp=C/'actual_corpus/ORIGINAL_WEAK_AND_COMPOSITE_LABELS4896_GT_ONLY.jsonl';ep=C/'input_preparation/CUES32_WEAK_EVIDENCE_GT_ONLY.jsonl'
 inputs=rows(ip);gt=rows(gp);evidence=rows(ep);assert len(inputs)==len(gt)==4896 and len(evidence)==32
 # Exact original component support native cell footprints; never shrunk/recentered.
 supports=[];reference=[]
 for ci,e in enumerate(evidence):
  assert e['cue_index']==ci;boxes=[]
  for r in e['original_regions']:
   b=np.array(r['bbox'],float);lo=np.floor(b[:2]+.5).astype(int)
   hi=np.ceil(b[2:]+.5).astype(int)if r['kind']=='HALF_OPEN_PIXEL_EDGE'else np.floor(b[2:]+.5).astype(int)+1
   boxes.append(np.r_[lo-.5,hi-.5])
  ref=np.array(e['reference_xy'],float);rp=np.floor(ref+.5).astype(int);boxes.append(np.r_[rp-.5,rp+.5]);supports.append(np.array(boxes));reference.append(ref)
 pairs=[];cache={}
 for ci in range(32):
  # Existing centre offset0 is read only, not a replacement for any original initializer q.
  base=inputs[(9*ci+4)*17];assert np.array_equal(base['query_xy'],reference[ci]);L=float(base['L']);H,W=base['raw_hw']
  for rank in range(8):
   i=(9*ci+4)*17+1+2*rank;c,s=inputs[i],inputs[i+1];cg,sg=gt[i],gt[i+1]
   assert c['cue_index']==s['cue_index']==ci and c['donor_rank']==s['donor_rank']==rank
   if not cg['geometry_pair_ready']:assert not sg['geometry_pair_ready'];continue
   assert (cg['candidate_aux_hidden_label_GT_only'],sg['candidate_aux_hidden_label_GT_only'])==(1,0)
   cr,sr=c['full_RGB_recipe'],s['full_RGB_recipe'];assert cr['patch']==sr['patch'];f=Path(cr['patch']['path'])
   if str(f)not in cache:
    assert bind(f)==cr['patch']
    with np.load(f,allow_pickle=False)as a:alpha=np.array(a['alpha'],copy=True)
    assert alpha.dtype==bool and alpha.ndim==2;cache[str(f)]=alpha
   pairs.append(dict(ci=ci,rank=rank,L=L,HW=(H,W),source=base['source'],raw_split=base['raw_split'],alpha=cache[str(f)],cover_xy=cr['placement_xy'],sham_xy=sr['placement_xy']))
 assert len(pairs)==230
 def inside(alpha,xy,p):
  x,y=p[0]-xy[0],p[1]-xy[1];return bool(alpha[y,x])if 0<=x<alpha.shape[1]and 0<=y<alpha.shape[0]else False
 results={}
 for amp in AMPS:
  total=Counter();bygroup=defaultdict(Counter)
  for pair in pairs:
   ci=pair['ci'];H,W=pair['HW'];b=supports[ci];half=pair['L']/2;group=pair['source']+'::'+pair['raw_split']
   for dy in(-amp,0,amp):
    for dx in(-amp,0,amp):
     q=reference[ci]+[dx,dy];p=np.floor(q+.5).astype(int);valid=0<=q[0]<W and 0<=q[1]<H and 0<=p[0]<W and 0<=p[1]<H
     native_all=bool((b[:,:2]>=0).all()and(b[:,2:]<=np.array([W,H])).all())
     full=bool((b[:,:2]>=q-half).all()and(b[:,2:]<=q+half).all())
     c=inside(pair['alpha'],pair['cover_xy'],p)if valid else None;s=inside(pair['alpha'],pair['sham_xy'],p)if valid else None
     context_changed=bool((q<np.array(base['target_bbox_xyxy'][:2])).any()) if False else False
     keys=['all_fixed_offsets']
     keys+=['q_native_valid'if valid else'q_native_OOB','uncertainty_FULL_native_crop'if full else'uncertainty_PARTIAL_or_MISS_native_crop']
     if valid:
      keys+=['COVER_q_inside'if c else'COVER_q_outside','SHAM_q_inside'if s else'SHAM_q_outside']
      keys+=['joint_C'+str(int(c))+'_S'+str(int(s))]
      if full and native_all:
       keys+=['q_valid_and_entire_original_uncertainty_FULL','FULL_joint_C'+str(int(c))+'_S'+str(int(s))]
       if not c:keys+=['FULL_COVER_semantic_E_q_alpha_outside']
       if s:keys+=['FULL_SHAM_semantic_D_q_alpha_inside']
     for k in keys:total[k]+=1;bygroup[group][k]+=1
  results[str(amp)]=dict(counts=dict(total),by_source_rawsplit={k:dict(v)for k,v in bygroup.items()})
 v93=P/'query_visibility_synthetic_part_teacher_20261023_v93/ROOT_PHASE_RESULTS_KO.json';v95=P/'query_visibility_same_q_counterfactual_20261023_v95/ROOT_RESULTS_KO.json';v95scope=v95.parent/'ROOT_SOURCE_SCOPE_BEFORE_ACTUAL_CONSTRUCTION.json';v95q=v95.parent/'source_diagnosis/QUERY_TO_AUX_SUPPORT_METADATA.json';prior=json.loads(v95q.read_text())
 result=dict(completed=True,scope='Read-only geometry feasibility, not new inputs/teacher labels/admission or selected model.',axis_amplitudes_px=list(AMPS),offset_rule='For each amplitude use all9 Cartesian offsets {-a,0,a}^2 around original weakreference; keep original L0 and exact same native cover/sham patches/placements; no alpha-best q selection.',existing_assets_previously_exposed=True,inputs=bind(ip),GT_sidecar=bind(gp),original_weak_evidence=bind(ep),producer_code=bind(C/'composite_preparation/build_flat_cue_corpus.py'),references=[bind(v93),bind(v95),bind(v95scope),bind(v95q)],pairs230_shared_per_amplitude=True,failed26_cue_ranks_not_replaced=True,original_uncertainty_footprints='Exact V122 native-cell endpoints derived from original HALF_OPEN or CLOSED ranges and reference cell; FULL checks whole footprint, no shrinking.',feasibility=results,duplicate_evidence=dict(V93='Wholepart/ownership diagnostic at posthoc AABBcentre, fixedL64; semanticDES UNKNOWN. No q-offset factorial or training.',V95=dict(accepted_pairs=15,cover_q_inside=prior['accepted_pairs']['cover_nearest_q_inside'],cover_q_outside=15-prior['accepted_pairs']['cover_nearest_q_inside'],sham_q_inside=prior['accepted_pairs']['sham_nearest_q_inside'],sham_q_outside=15-prior['accepted_pairs']['sham_nearest_q_inside'],meaning='Already semantic recorded-cue visibility versus q-alpha separation at original q; small clustered Pallox set, not controlled within-same-image q factorial.')),new_input_manifest_written=False,new_teacher_labels_written=False,new_training_admission=False,native_mapping='UNKNOWN',human_gold=False,actual_RGB_model_fit_current_V112_PnP=0,script=bind(Path(__file__)))
 output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'report':bind(output),'by_amplitude':{k:v['counts']for k,v in results.items()}},ensure_ascii=False))
if __name__=='__main__':main()
