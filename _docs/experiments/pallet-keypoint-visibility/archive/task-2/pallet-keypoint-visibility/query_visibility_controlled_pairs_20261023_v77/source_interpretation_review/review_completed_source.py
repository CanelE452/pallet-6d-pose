"""Saved source-only methodological review; NumPy/JSON only, no model execution."""
from pathlib import Path
import csv,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'runs/PAIRED_SUPPORT_V1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def npz(p,keys):
    with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in keys}
def rate(n,d):return float(n/d)if d else None
def write_new(p,x):
    with open(p,'x')as f:f.write(json.dumps(x,indent=2)+'\n')

assert sha(RUN/'FIT_COMPLETE.json')=='1af4ec060f19bde2dc70282f99d6274d98e3be8f6066c7f5b56f365647d6b823'
complete=load(RUN/'FIT_COMPLETE.json');assert complete['completed']and complete['optimizer_updates']==2000
for key in ('model_freeze','scores','source_results','history'):check(complete[key])
base_seal=load(ROOT/'frozen_V76_source_baseline/BASELINE_SEAL.json')
for key in ('entry','scores','results'):check(base_seal[key])
assert base_seal['PASS']and base_seal['original_logit_differences']==0
new=load(RUN/'SOURCE_RESULTS.json');old=load(ROOT/'frozen_V76_source_baseline/SOURCE_RESULTS.json')
keys=['support_logit','NLL_FP32','input_available','primary_loss_mask','reference_in_local_square_GT_only','id','source','split','domain','after_original_row_index','after_anchor_k','pair_good']
score={name:npz(path,keys)for name,path in [('V76',ROOT/'frozen_V76_source_baseline/SOURCE_SCORES.npz'),('V77',RUN/'SELECTED_SOURCE_SCORES.npz')]}
for k in keys[2:]:assert np.array_equal(score['V76'][k],score['V77'][k]),k
g0p=ROOT.parent/'query_visibility_reference_local_support_20261023_v76/gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz'
g1p=ROOT/'pair_generation/AFTER_GT_ONLY.npz'
g0=npz(g0p,['label8','id','source','split']);g1=npz(g1p,['label8','id','source','split','anchor_k','original_row_index','pair_anchor_success'])
assert sha(g0p)=='626338ae0dee48f5c5d5f53930e05d02960225ee9d3972e751aeff4b25c30f45'
N=5203;A=4799;y=np.concatenate([g0['label8'],g1['label8']]);s=score['V77']
for k in ('id','source','split'):assert np.array_equal(np.concatenate([g0[k],g1[k]]),s[k]),k
assert s['support_logit'].shape==(N+A,8)and s['primary_loss_mask'].shape==y.shape
assert not(s['primary_loss_mask']&(y<0)).any()
assert not(s['primary_loss_mask']&~s['input_available']).any()
assert not(s['primary_loss_mask']&~s['reference_in_local_square_GT_only']).any()
assert np.array_equal(s['after_anchor_k'],g1['anchor_k'])and np.array_equal(s['after_original_row_index'],g1['original_row_index'])
comparison=[]
for key,v in new['results']['natural_and_after'].items():
    b=old['results']['natural_and_after'][key];assert v['all_slots']==b['all_slots']and v['admitted']==b['admitted']and v['U']==b['U']
    row=dict(cohort=key,N=v['admitted'],all_slots=v['all_slots'],teacher_U=v['U'])
    for metric in ('visible_precision','visible_recall','covered_precision','covered_recall','mean_BCE_FP32'):
        row['V76_'+metric]=b['metrics'][metric];row['V77_'+metric]=v['metrics'][metric];row['delta_'+metric]=v['metrics'][metric]-b['metrics'][metric]
    row['V76_CM']=b['metrics']['confusion_rows_GT_cols_pred_COVERED_VISIBLE'];row['V77_CM']=v['metrics']['confusion_rows_GT_cols_pred_COVERED_VISIBLE']
    comparison.append(row)
with open(HERE/'SOURCE_METRIC_CHANGES.csv','x',newline='')as f:
    w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)

same=[];macro={};train_val=[]
for model,z in score.items():
    orig=z['after_original_row_index'];ak=z['after_anchor_k'];aidx=np.arange(A);after=z['support_logit'][N:];before=z['support_logit'][orig]
    anchorC=after[aidx,ak]<0;ap=z['primary_loss_mask'][N:];yp=y[N:];assert np.all(yp[aidx,ak]==0)
    nv=ap&(yp==1);nv[aidx,ak]=False;npred=after>=0
    for source in ('SYNTHETIC','YCB_PUBLIC_REAL'):
        for split in ('train','val'):
            rows=(z['source'][N:]==source)&(z['split'][N:]==split)
            v=nv&rows[:,None];mixed=rows&nv.any(1)
            cv=anchorC[:,None]&npred&v
            stable=v&(g0['label8'][orig]==1)&z['primary_loss_mask'][orig]
            same.append(dict(model=model,source=source,split=split,
                eligible_mixed_after_instances=int(mixed.sum()),known_V_nonanchor_queries=int(v.sum()),
                V_nonanchor_correct=int((npred&v).sum()),V_nonanchor_recall=rate(int((npred&v).sum()),int(v.sum())),
                C_anchor_and_V_nonanchor_both_correct_query_pairs=int(cv.sum()),same_after_both_rate=rate(int(cv.sum()),int(v.sum())),
                V_nonanchor_logit_gt_C_anchor_query_pairs=int(((after>after[aidx,ak,None])&v).sum()),
                all_V_nonanchors_and_C_anchor_correct_instances=int((mixed&anchorC&np.all(npred|~nv,axis=1)).sum()),
                stable_before_after_V_queries=int(stable.sum()),before_V_correct_after_wrong=int((stable&(before>=0)&~npred).sum()),before_V_wrong_after_correct=int((stable&(before<0)&npred).sum())))
    strata=[]
    for source in ('SYNTHETIC','YCB_PUBLIC_REAL'):
        for label in (0,1):
            mask=(z['source']==source)[:,None]&(z['split']=='val')[:,None]&(z['domain']==0)[:,None]&(y==label)&z['primary_loss_mask']
            strata.append(dict(source=source,label=label,N=int(mask.sum()),mean_BCE_FP32=float(z['NLL_FP32'][mask].mean(dtype=np.float64))))
    macro[model]=dict(value=float(np.mean([q['mean_BCE_FP32']for q in strata])),strata=strata)
for model,d in [('V76',old),('V77',new)]:
    for domain in ('NATURAL_ORIGINAL','CONTROLLED_AFTER'):
        for source in ('SYNTHETIC','YCB_PUBLIC_REAL'):
            tr=d['results']['natural_and_after'][f'{domain}|{source}|train']['metrics'];va=d['results']['natural_and_after'][f'{domain}|{source}|val']['metrics']
            train_val.append(dict(model=model,domain=domain,source=source,TRAIN=tr,VAL=va,VAL_minus_TRAIN_V_recall=va['visible_recall']-tr['visible_recall'],VAL_minus_TRAIN_C_recall=va['covered_recall']-tr['covered_recall'],VAL_minus_TRAIN_meanBCE=va['mean_BCE_FP32']-tr['mean_BCE_FP32']))
ind=load(ROOT/'independent_saved_audit/INDEPENDENT_SELECTED_SAVED_AUDIT.json')
for row in same:
    if row['model']!='V77':continue
    ref=next(q for q in ind['paired_and_same_after_control_diagnostics']if q['source']==row['source']and q['split']==row['split'])
    assert row['known_V_nonanchor_queries']==ref['same_after_C_anchor_V_nonanchor_query_pairs']
    assert row['C_anchor_and_V_nonanchor_both_correct_query_pairs']==ref['same_after_both_correct']
    assert row['before_V_correct_after_wrong']==ref['nonanchor_V_before_correct_after_wrong']
    assert row['before_V_wrong_after_correct']==ref['nonanchor_V_before_wrong_after_correct']

review=dict(
 completed=True,PASS=True,PASS_meaning='Saved-source comparison and methodological review completed; not candidate admission or a solution claim.',
 status='PARTIAL_CONTROLLED_PAIR_GAIN_WITH_NATURAL_VISIBLE_PRESERVATION_REGRESSION',fully_solved=False,
 artifact_bindings=[bind(RUN/'FIT_COMPLETE.json'),bind(RUN/'MODEL_FREEZE.json'),bind(RUN/'SOURCE_RESULTS.json'),bind(RUN/'SELECTED_SOURCE_SCORES.npz'),bind(ROOT/'frozen_V76_source_baseline/BASELINE_SEAL.json'),bind(ROOT/'frozen_V76_source_baseline/SOURCE_RESULTS.json'),bind(ROOT/'frozen_V76_source_baseline/SOURCE_SCORES.npz'),bind(ROOT/'independent_saved_audit/INDEPENDENT_SELECTED_SAVED_AUDIT.json'),bind(HERE/'METADATA_AND_REVIEW_PLAN_BEFORE_RESULTS.json'),bind(g0p),bind(g1p)],
 naturalVAL4class_macroBCE=macro,naturalVAL_macro_delta=macro['V77']['value']-macro['V76']['value'],
 natural_and_after_metric_changes=comparison,paired_anchors={name:d['results']['paired_anchors']for name,d in [('V76',old),('V77',new)]},
 same_after_query_control_comparison=same,TRAIN_VAL_comparisons=train_val,
 findings_KO=[
  'V77은 고정 source 가림 개입에 대한 반응을 개선했다. SYNTH VAL pair 양쪽 정답은127/230→220/230, 공개 YCB는133/731→268/731이다. 그러나 공개 YCB에서 새 모델도463/731 pair는 양쪽 모두 맞히지 못한다.',
  '자연 YCB VAL에서는 COVERED 회수173→202/287과 visible precision 개선을 얻었지만, 실제 visible 회수는1157→1080/1731로77개 감소했다. SYNTH natural VAL V 회수도1422→1404/1452로18개 감소했다. 가림 반응 개선을 보이는 점 보존 성공으로 해석할 수 없다.',
  'YCB after의 known V 비-anchor는669→617/948로52개 줄었다. 다만 V77 자체의 같은 V 슬롯 전후 오분류 이동은18개 정답→오답,20개 오답→정답으로 거의 균형이다. 따라서 V76 대비 저하는 붙임 직후의 전역붕괴만이 아니라 새 학습의 전반적 visible 판별 약화도 포함한다.',
  'YCB TRAIN pair 양쪽 정답도1371/3095(44.3%)로 낮고 VAL은268/731(36.7%)이다. held-out 과적합만으로 전체 실패를 설명하기 어렵다. SYNTH pair는 TRAIN94.8%/VAL95.7%로 안정적이나 같은 증거를 공개 실사 일반화로 옮길 수 없다.',
  '동일 after RGB에서 C anchor와 V nonanchor를 구분한 경우는 단일 이미지 전체를 무조건 COVERED로 부르는 상수 규칙보다 강한 증거다. 그러나 local RGB/P3, q 위치, context의 native index 또는 전체 target context 중 어느 입력을 사용했는지는 이 비교만으로 식별되지 않는다.',
  '고정 q/L/box의 RGB 전후 비교는 query 좌표 jitter나 crop 이동 민감도를 측정하지 않는다. GT reference 위치가 입력에 숨겨진 semantic 식별 문제를 해결했다는 증거도 아니다.',
  '동일 opaque cutout/경계/배율 규칙이 TRAIN과 VAL에 사용되어 합성 seam·붙임 모양 shortcut을 배제하지 못한다. 자연 VAL 보존 검사와 같은 이미지 비-anchor V 대조는 그 위험을 일부 견제하지만 실제 자연 가림의 별도 검증을 대신하지 않는다.',
  '실제 donor instance/RGB와 recipient RGB는 split 사이 중복0이고 public scene0..7/8..9를 지킨다. 그러나 donor object ID4,5,9,12,15,16은 양쪽 split에 있어 물체 종류/물리 인스턴스 전체 hold-out이라고 주장할 수 없다. VAL donor ID6,11은 TRAIN에서 보이지 않지만 혼합된 전체 성능으로 그 부분의 일반화를 입증할 수는 없다.',
  '자연 VAL은 checkpoint를 고르는 동일 집합이다. controlled VAL은 selector에 미사용이지만 같은 자연 VAL 장면과 동일 generator를 공유한다. 공개 query는 emulated source initial points이며 현재 native initializer end-to-end 검증이 아니다.',
 ],
 provisional_DINO_representation_review=dict(
  status='PROPOSAL_REVIEW_ONLY_NO_EXECUTION_OR_SELECTION',
  output_contract='Fixed support scalar logit at the unchanged initial query; fixed .5. Negative remains source reference-local COVERED support, not semantic E/O/S. U/missing admission unchanged.',
  suggested_feature_accounting='384 centre2x2 local +384 valid-masked local mean +384 valid-masked target mean =1152 features/query; plus context42 only if explicitly retained before protocol. A 768-input head requires declaring which vector/difference is dropped.',
  storage_accounting='For80000 queries FP16 two local384-vectors cost122880000bytes. Deduplicated target384-vector per~10002 frame costs7681536bytes. Repeating all three per query would instead cost184320000bytes.',
  crop_limit='Resizing cached local96/target128 to224 does not recover absent native content, fix q errors, or make a missed reference visible. Full query crop spanL and target crop geometry remain the factual observation limit.',
  token_validity='Freeze interpolation/normalization/native orientation and validity→16x16 token weights, handling partial-token support/zero mass before fit. Validity is input geometry, not GT foreground ownership.',
  isolation_limit='Replacing P3/RGB encoders by pooled embeddings changes representation, pooling, spatial resolution and head parameterization. A gain is a pipeline-component comparison unless common data/sampler/teacher/seed/budget/selector/threshold and an explicit pooling/head control isolate effects.',
  identity_limit='A generic target mean is a target-context cue, not a certified target-owned pixel mask or semantic corner dictionary. Do not introduce reference, mask, source ID, donor ID or GT-native correspondence into forward.',
  no_claim='No extraction/fit/current evaluation authorized or executed by this review; viability and benefit remain hypotheses.'),
 source_GT_labels_and_saved_scores_read=True,RGB_masks_features_weights_current_reads=0,new_model_backbone_forward_fit_download_PnP=0,
 code=bind(Path(__file__)),metric_changes_csv=bind(HERE/'SOURCE_METRIC_CHANGES.csv'),
)
write_new(HERE/'SOURCE_INTERPRETATION_REVIEW.json',review)
write_new(HERE/'REVIEW_SEAL.json',dict(completed=True,PASS=True,artifacts=[bind(HERE/'SOURCE_INTERPRETATION_REVIEW.json'),bind(HERE/'SOURCE_METRIC_CHANGES.csv'),bind(HERE/'review_completed_source.py'),bind(HERE/'METADATA_AND_REVIEW_PLAN_BEFORE_RESULTS.json')],current_forward_fit_PnP=0))
print(json.dumps(bind(HERE/'SOURCE_INTERPRETATION_REVIEW.json')))
