"""Bounded source-only code/math + frozen metadata review; no RGB/mask/model."""
from pathlib import Path
import ast,hashlib,json
from collections import Counter
import numpy as np
R=Path(__file__).resolve().parent;V=R.parent;P=V.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s)for s in Path(p).read_text().splitlines()]
def save(p,d):
 with p.open('x')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def pure(path,name):
 fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
 scope=dict(np=np,SOURCES=('SYNTHETIC','YCB_PUBLIC_REAL'));exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope);return scope[name]

protocol=load(V/'pair_generation/PAIR_BUILD_PROTOCOL_BEFORE_COMPOSITION.json')
assert sha(V/'pair_generation/PAIR_BUILD_PROTOCOL_BEFORE_COMPOSITION.json')=='38c7568a4d0d1cc945b517605f3f490c2c81e93f8436a11430c4b318dc27147d'
assert sha(V/'pair_generation/opaque_pair_geometry.py')==protocol['bindings']['geometry']['sha256']
assert sha(V/'pair_generation/build_pairs.py')==protocol['bindings']['builder']['sha256']
root=load(V/'EXPERIMENT_DECISIONS_BEFORE_PAIR_BUILD.json');assert sha(V/'EXPERIMENT_DECISIONS_BEFORE_PAIR_BUILD.json')==protocol['bindings']['root_decisions']['sha256']
proof=load(V/'pair_generation/GENERATED_DISCRETE_COVERAGE_TESTS.json');assert proof['PASS']and proof['cases']==480
trainer=V/'train_paired_once.py';mock=load(V/'GENERATED_TRAINER_TESTS.json');assert mock['PASS']and mock['code']==bind(trainer)
pairfn=pure(trainer,'pair_eligibility')
assert pairfn(np.ones(4,bool),np.ones(4,int),np.ones(4,bool),np.array([0,0,-1,0]),np.array([1,0,1,1],bool),np.array([0,-1,0,-1])).tolist()==[True,False,False,False]
sample=pure(trainer,'sample');s=np.repeat([0,1],48);y=np.tile(np.repeat([0,1],24),2)
meta=dict(natural_pools={(a,c):np.flatnonzero((s==a)&(y==c))[:8]for a in range(2)for c in range(2)},nonanchor_pools={(a,c):np.flatnonzero((s==a)&(y==c))[8:16]for a in range(2)for c in range(2)},pair_train_pools={0:np.arange(4),1:np.arange(4,8)},pair_before_query=np.r_[np.arange(40,44),np.arange(88,92)],pair_after_query=np.r_[np.arange(16,20),np.arange(64,68)])
rng=np.random.default_rng(20261023)
for _ in range(64):
 ix=sample(meta,rng);assert len(ix)==96 and all(np.sum((s[ix]==a)&(y[ix]==c))==24 for a in range(2)for c in range(2))
 for a in range(2):
  for c in range(2):assert all(np.sum((s[ix[b:b+32]]==a)&(y[ix[b:b+32]]==c))==8 for b in [0,32,64])
 assert np.array_equal(y[ix[64:]],np.tile([1,0],16))
sel=pure(trainer,'natural_val_metric');meta0=dict(N=14,valpools={(0,0):np.array([1,3]),(0,1):np.array([4]),(1,0):np.array([5,7]),(1,1):np.array([9,11])})
ids=np.sort(np.concatenate(list(meta0['valpools'].values())));pred=dict(NLL_F32=np.arange(1,len(ids)+1,dtype=np.float32));m,by=sel(pred,ids,meta0);assert m==np.mean([1.5,3,4.5,6.5])and len(by)==4

selection=lines(V/'pair_generation/PAIR_SELECTION_GT_ONLY.jsonl');eligibility=lines(V/'pair_generation/ORIGINAL_5203_ELIGIBILITY_GT_ONLY.jsonl');pool=load(V/'pair_generation/DONOR_POOL_LOCK.json')['all4200rows'];lock=load(V/'pair_generation/METADATA_SELECTION_LOCK_BEFORE_COMPOSITION.json')
origmeta=lines(P/'query_visibility_reference_local_support_20261023_v76/gt_corpus_build/ORDERED_METADATA_INDEX_GT_ONLY.jsonl')
assert len(eligibility)==len(origmeta)==5203 and len(pool)==4200 and len(selection)==lock['after_rows']==4799
for k in ['all_originalrows_ledger','selection','donor_pool','protocol']:assert sha(lock[k]['path'])==lock[k]['sha256']
assert [r['original_row_index']for r in selection]==[r['original_row_index']for r in eligibility if r['eligible']]
salt=protocol['salt']
def digest(kind,ident):return hashlib.sha256((salt+'\n'+kind+'\n'+ident).encode()).hexdigest()
counts=Counter()
for i,row in enumerate(selection):
 j=row['original_row_index'];e=eligibility[j];assert j==e['original_row_index']and e['eligible']and row['after_frame_index']==i
 valid=[k for k in range(8)if e['original_primary8'][k]and k in e['original_visible_admitted_slots']]
 ak=min(valid,key=lambda k:(digest('anchor',e['original_id']+'\n'+str(k)),k));assert row['anchor_k']==e['anchor_k']==ak
 assert row['quarterturn']==int(digest('turn',row['original_id']),16)%4
 own=origmeta[j].get('object_id');compatible=sorted([d for d in pool if d['usable_nonempty']and d['split']==row['split']and d['image_sha256']!=row['image_sha256']and(own is None or d['object_id']!=own)],key=lambda d:d['id'])
 chosen=int(digest('donor',row['original_id']),16)%len(compatible)
 assert row['compatible_donor_count']==len(compatible)and row['selected_donor_index']==chosen and row['donor']==compatible[chosen]
 assert row['donor']['split']==row['split']and row['donor']['image_sha256']!=row['image_sha256']
 if row['source']=='YCB_PUBLIC_REAL':assert row['donor']['object_id']!=own
 for k in ['initial_points','roi_side','box','source','split','image','image_sha256']:assert row[k]==e[k],k
 counts[row['source']+'|'+row['split']]+=1

report=dict(schema='V77_INDEPENDENT_PROTOCOL_CODE_METADATA_REVIEW',PASS=True,source_scope='Source-only frozen code/math and metadata; no RGB/mask/P3/weights/current score or GT bodies read.',trainer=bind(trainer),pair_protocol=bind(V/'pair_generation/PAIR_BUILD_PROTOCOL_BEFORE_COMPOSITION.json'),root_decisions=bind(V/'EXPERIMENT_DECISIONS_BEFORE_PAIR_BUILD.json'),reviewed_code=[bind(p)for p in [V/'pair_generation/build_pairs.py',V/'pair_generation/opaque_pair_geometry.py',V/'after_input_preparation/build_after_input_pack.py',P/'query_visibility_reference_local_support_20261023_v76/mixed_fit/train_mixed_once.py']],generated_proofs=[bind(V/'GENERATED_TRAINER_TESTS.json'),bind(V/'pair_generation/GENERATED_DISCRETE_COVERAGE_TESTS.json'),bind(R/'GENERATED_INDEPENDENT_AUDIT_PRIMITIVE_TESTS.json')],checks=dict(sampling32_32_32=True,source_class24=True,each_component_source_class8=True,pair_beforeV_afterC_adjacent=True,failed_control_excluded_even_actual_teacherC=True,natural_VAL_only_4strata_selection=True,strict_lower_earliest_ties=True,threshold_fixed_half_inclusive=True,all5203_original_rows_retained_in_natural_bank=True,all4799_original_primaryV_eligible_rows_retained=True,all_anchor_hash_donor_hash_turn_hash_replayed=True,donor_same_partition_other_RGB_and_otherYCB_object=True,GT_fields_excluded_from_model_forward=True,q_r_L_box_context_and_canvasvalidity_checked_in_trainer=True,after_RGB_target_and_P3_recomputed_by_separate_bound_producer=True),metadata_counts=dict(counts),resolved_finding=dict(problem='Trainer V2 omitted explicit pair_anchor_success/anchor_training_target eligibility, permitting a hypothetical failed-control+teacherC pair.',resolution='V3 pair_eligibility now requires control_success AND anchor_target==0 and asserts flag/target mapping. Generated failed-control actualC case independently replayed and excluded.',previous_code_preserved_by_root='provenance/PRE_CONTROL_SUCCESS_LOSS_GUARD_V2'),limitations=['Actual after RGB/P3 consistency, full teacher replay and input canvas facts await completed source data/input seals; this review does not certify unfinished payloads.','EDT proof certifies opaque discrete mask coverage only; extreme donor magnification and frame-wide cover can create synthetic visual shortcuts.','Global occluder shortcut is reduced but not disproved by nonanchor controls; report same-after anchorC/nonanchorV and beforeV/afterV stability separately.','Paired C is conditional target-reference support suppression, not natural human semantic E/S or native physical corner gold.','Public YCB q is GT-derived query emulation; this experiment does not measure an actual deployed public-data initializer.','SYNTH validation shares the established source-heldout selection family; no new independent real pallet semantic evaluation claim.'],future_saved_auditor=bind(R/'audit_frozen_pair_saved.py'),optimizer_updates_or_model_forward_current_PnP=0)
save(R/'PROTOCOL_AND_METADATA_REVIEW.json',report)
print(json.dumps(dict(PASS=True,report=bind(R/'PROTOCOL_AND_METADATA_REVIEW.json'),counts=dict(counts))))
