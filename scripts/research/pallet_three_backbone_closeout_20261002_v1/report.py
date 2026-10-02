"""CPU-only closeout of already sealed results; no inference, weights or raw caches."""
from pathlib import Path
import argparse, csv, hashlib, json, math, os, re, statistics
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments' / HERE.name
EXP = ROOT / '_docs/experiments'
Y = 'pallet_sensors_submission_v1'
D = 'pallet_dope_refiner_20261001_v1'
R = 'pallet_resnet18_dim_refiner_20261002_v1'
V = 'pallet_resnet18_dim_refiner_report_20261002_v3'
B = 'pallet_resnet18_dimension_20261002_v1'
T = 'pallet_three_backbone_runtime_20261002_v1'
S = 'pallet_green0918_dimension_audit_v1'
SOURCES = {
 'yolo':f'{Y}/UNIFIED_DEV_RESULTS.json', 'yolo_report':f'{Y}/FINAL_REPORT_KO.md',
 'yolo_status':f'{Y}/FINAL_STATUS.json', 'yolo_receipt':f'{Y}/EVALUATE_COMPLETE.json',
 'yolo_prior':f'{Y}/P_VS_PRIOR_PAIRED.json',
 'dope':f'{D}/DEV_RESULTS.json', 'dope_paired':f'{D}/DEV_PAIRED_RESULTS.json',
 'dope_report':f'{D}/REPORT_KO.md', 'dope_receipt':f'{D}/REPORT_COMPLETE.json',
 'dope_gallery':f'{D}/GALLERY_MANIFEST.json',
 'resnet':f'{R}/DEV_RESULTS.json', 'resnet_paired':f'{R}/DEV_PAIRED_RESULTS.json',
 'resnet_tables':f'{V}/AGGREGATE_TABLES.json', 'resnet_report':f'{V}/REPORT_KO.md',
 'resnet_receipt':f'{V}/REPORT_MANIFEST.json', 'claim_audit':f'{V}/CLAIM_AUDIT.json',
 'direct_report':f'{B}/REPORT_KO_V2.md', 'direct_receipt':f'{B}/DSNT_REPORT_MANIFEST_V2.json',
 'direct_gallery':f'{B}/DSNT_GALLERY_MANIFEST.json', 'direct_gallery_md':f'{B}/GALLERY.md',
 'runtime':f'{T}/RESULTS.json','runtime_protocol':f'{T}/PROTOCOL.json',
 'runtime_receipt':f'{T}/REPORT_COMPLETE.json','runtime_report':f'{T}/REPORT_KO.md',
 'square_report':f'{S}/REPORT_KO.md', 'square_snapshot':f'{S}/DATASET_SNAPSHOT.json',
}
for arm in ('CONSTANT','SHAPE','FULL'):
 SOURCES['direct_'+arm] = f'{B}/DSNT_FULL_{arm}_REAL_RESULTS.json'
 SOURCES['direct_pose_'+arm] = f'{B}/DSNT_FULL_{arm}_POSE_RESULTS.json'
IMAGES = [f'{D}/figures/case_01.png',f'{D}/figures/case_04.png',
 f'{B}/DSNT_GALLERY/GREEN0918_119_IMPROVEMENT.png', f'{B}/DSNT_GALLERY/GREEN0918_119_WORSENING.png']
for i,p in enumerate(IMAGES): SOURCES[f'rgb_{i}'] = p

def read(p): return json.loads(Path(p).read_text())
def sha(p):
 p=Path(p); assert p.stat().st_size < 20_000_000, ('Large raw/checkpoint read forbidden',p)
 return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p):
 p=Path(p).resolve(); assert p.is_relative_to(ROOT)
 return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
def verify_binding(b):
 p=(ROOT/b['path']).resolve(); assert p.is_relative_to(ROOT)
 assert sha(p)==b['sha256'], ('SHA mismatch',b['path'])
 if 'bytes' in b: assert p.stat().st_size==b['bytes'], ('size mismatch',b['path'])
 return p
def write(name,value):
 p=(DOC/name).resolve(); assert p.is_relative_to(DOC)
 p.parent.mkdir(parents=True,exist_ok=True)
 text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
 p.write_text(text)
def bindings_in(o):
 if isinstance(o,dict):
  if isinstance(o.get('path'),str) and isinstance(o.get('sha256'),str): yield o
  for v in o.values(): yield from bindings_in(v)
 elif isinstance(o,list):
  for v in o: yield from bindings_in(v)
def seal():
 pin=dict(schema='three_backbone_closeout_source_pin_v1',inputs={k:bind(EXP/v) for k,v in SOURCES.items()},
  scope='Existing public result JSON/MD and existing rendered RGB only. No checkpoints/raw caches.',
  new_fits=0,new_forwards=0,new_pose_solves=0)
 p=DOC/'SOURCE_BINDINGS.json'
 if p.exists(): assert read(p)==pin, 'Frozen source pin changed'
 else: write(p.name,pin)
 return pin

def load():
 pin=read(DOC/'SOURCE_BINDINGS.json'); assert set(pin['inputs'])==set(SOURCES)
 docs={}
 for k,b in pin['inputs'].items():
  assert b['path']==str((EXP/SOURCES[k]).relative_to(ROOT))
  p=verify_binding(b)
  if p.suffix=='.json': docs[k]=read(p)
 selected={b['path']:b for b in pin['inputs'].values()}
 edges=[]
 for owner,obj in docs.items():
  for b in bindings_in(obj):
   if b['path'] in selected:
    assert b['sha256']==selected[b['path']]['sha256'], ('Historical receipt mismatch',owner,b['path'])
    verify_binding(b);edges.append(dict(receipt=owner,target=b['path'],sha256=b['sha256']))
 for k in ['yolo','dope','resnet','resnet_tables','resnet_receipt','direct_receipt','runtime','runtime_receipt']:
  assert docs[k]['complete'] is True,k
 assert docs['runtime']['PASS'] and docs['runtime']['all_2d_parity_PASS'] and docs['runtime']['all_pose_parity_PASS']
 assert docs['yolo']['role']==docs['dope']['role']==docs['resnet']['role']=='REUSED_DEV'
 assert len(edges)>=20
 return docs,pin,edges

def number_row(backbone,method,stats,source,selectors,seed_count,main):
 p=stats['pose'];frames=319
 return dict(backbone=backbone,method=method,seed_count=seed_count,main_comparison=main,
  population='REUSED_DEV319',frames=frames,sessions=13,gt_landmarks=2818,
  matched_frames=stats['matched_frames'],conditional_landmarks=stats['supervised_points'],
  median_px=stats['median_px'],p90_px=stats['p90_px'],all_gt_pck10=stats['ALL_GT_PCK']['10'],
  translation_median_cm=p['translation_median_cm'],rotation_median_deg=p['rotation_median_deg'],
  pose_coverage=p['coverage'],source_json=source,source_selectors=selectors,
  aggregation='single_frozen_baseline' if seed_count==1 else 'mean_of_three_seed_statistics_not_ensemble')
def average_stats(rows):
 keys=['median_px','p90_px','matched_frames','supervised_points']
 result={k:statistics.mean(r[k] for r in rows) for k in keys}
 result['ALL_GT_PCK']={'10':statistics.mean(r['ALL_GT_PCK']['10'] for r in rows)}
 result['pose']={k:statistics.mean(r['pose'][k] for r in rows) for k in ['translation_median_cm','rotation_median_deg','coverage']}
 return result

def summarize(d):
 accuracy=[]
 for backbone,key,baseline,arms in [('YOLO','yolo','R0',['D','P']),('DOPE','dope','DOPE',['D','P']),('ResNet18','resnet','FULL',['D0','P0','P5','P5_CONSTANT'])]:
  methods=d[key]['methods'];main_arm='P0' if key=='resnet' else 'P'
  accuracy.append(number_row(backbone,baseline,methods[baseline],SOURCES[key],['methods',baseline],1,True))
  for arm in arms:
   names=[f'{arm}_S{s}' if key=='resnet' else f'{arm}{s}' for s in (1,2,3)]
   stats=average_stats([methods[n] for n in names])
   if 'seed_mean' in d[key]:
    for k in ['median_px','p90_px','matched_frames','supervised_points']:
     assert math.isclose(stats[k],d[key]['seed_mean'][arm][k],rel_tol=1e-14,abs_tol=1e-14)
   accuracy.append(number_row(backbone,arm,stats,SOURCES[key],[['methods',n] for n in names],3,arm==main_arm))
 assert len(accuracy)==11
 for row in accuracy:
  assert row['gt_landmarks']==2818 and row['matched_frames']<=319
 # The separate V3 report must independently reproduce the selected ResNet numbers.
 for row in [r for r in accuracy if r['backbone']=='ResNet18']:
  other=next(x for x in d['resnet_tables']['arm_mean_rows'] if x['arm']==row['method'])
  for a,b in [('median_px','median_px'),('p90_px','p90_px'),('all_gt_pck10','PCK10'),('pose_coverage','pose_coverage'),('translation_median_cm','translation_median_cm'),('rotation_median_deg','rotation_median_deg')]:
   assert math.isclose(row[a],other[b],abs_tol=1e-13,rel_tol=1e-13)
 rt=d['runtime'];assert (rt['frames'],rt['sessions'],rt['repeats'],rt['batch'],rt['measured_calls'])==(26,13,5,1,1170)
 runtime=[]
 for arm,x in rt['summary'].items():
  runtime.append(dict(arm=arm,primary=arm in rt['primary_arms'],frames=26,repeats=5,calls=x['full_ms']['n'],
   two_d_median_ms=x['two_d_ms']['median'],two_d_p90_ms=x['two_d_ms']['p90'],pnp_median_ms=x['pnp_ms']['median'],
   full_median_ms=x['full_ms']['median'],full_p90_ms=x['full_ms']['p90'],pose_status_counts=x['pose_status_counts']))
 overhead=[]
 for backbone,base,corrected in [('YOLO','YOLO_R0','YOLO_P1'),('DOPE','DOPE_BASE','DOPE_P1'),('ResNet18','RESNET_FULL','RESNET_P0_S1')]:
  a=rt['summary'][base];b=rt['summary'][corrected]
  assert b['paired_baseline']==base
  assert b['paired_added_two_d_ms']['n']==b['paired_added_full_ms']['n']==130
  overhead.append(dict(backbone=backbone,baseline=base,correction=corrected,
   baseline_two_d_median_ms=a['two_d_ms']['median'],correction_two_d_median_ms=b['two_d_ms']['median'],
   delta_two_d_median_ms=b['two_d_ms']['median']-a['two_d_ms']['median'],
   baseline_full_median_ms=a['full_ms']['median'],correction_full_median_ms=b['full_ms']['median'],
   delta_full_median_ms=b['full_ms']['median']-a['full_ms']['median'],
   full_median_ratio=b['full_ms']['median']/a['full_ms']['median'],
   paired_added_two_d_median_ms=b['paired_added_two_d_ms']['median'],
   paired_added_full_median_ms=b['paired_added_full_ms']['median'],
   paired_added_two_d_p90_ms=b['paired_added_two_d_ms']['p90'],
   paired_added_full_p90_ms=b['paired_added_full_ms']['p90'],
   paired_observations=130,primary_overhead='median_of_same_frame_same_repeat_correction_minus_baseline',
   difference_definition='difference_of_marginal_medians_not_median_of_paired_differences'))
 direct=[];direct_pose=[]
 for arm in ['CONSTANT','SHAPE','FULL']:
  item=d['direct_'+arm]
  assert item['complete'] and item['arm']==arm and not item['independent_confirmation']
  for population in ['DEV319','GREEN150_MANUAL_DECLARED','GREEN0918_119_MANUAL_DECLARED']:
   x=item['results'][population]['summary']
   direct.append(dict(arm=arm,population=population,frames=x['total_frames'],matched_frames=x['matched'],
    conditional_corner_landmarks=x['observed_corners'],declared_corner_landmarks=x['corners'],
    median_corner8_px=x['matched_pooled_corner8_median_px'],p90_corner8_px=x['matched_pooled_corner8_P90_px'],
    all_gt_pck10=x['PCK']['10'],E_sym=x['E_sym'],source_json=SOURCES['direct_'+arm]))
  p=d['direct_pose_'+arm]['summary'];direct_pose.append(dict(arm=arm,frames=p['frames'],available=p['available'],
   translation_median_cm=p['translation_cm']['median'],rotation_median_deg=p['rotation_deg']['median'],
   coverage=p['coverage'],source_json=SOURCES['direct_pose_'+arm]))
 comparisons=[]
 def contrast(backbone,name,metric,obj):
  assert obj.get('status','COMPLETE')=='COMPLETE'
  return dict(backbone=backbone,contrast=name,metric=metric,status=obj.get('status','COMPLETE'),delta=obj['delta'],ci95_low=obj['low'],ci95_high=obj['high'],units=obj['units'],resamples=obj['resamples'],role='REUSED_DEV_exploratory_unadjusted')
 comparisons.append(contrast('YOLO','P_minus_R0','conditional_2D_median_px',d['yolo_status']['EVIDENCE']['P_minus_R0']['session']))
 comparisons.append(contrast('DOPE','P_minus_DOPE','conditional_2D_median_px',d['dope_paired']['results']['P_minus_DOPE']['conditional_keypoint_median']['session']))
 for x in d['resnet_tables']['paired_session_rows']:
  if x['contrast'] in ['P0_minus_FULL','P5_minus_FULL','P5_minus_P0','P5_minus_P5_CONSTANT'] and x['metric'] in ['conditional_2D_median_px','ALL_GT_PCK10','translation_median_cm','rotation_median_deg']:
   comparisons.append(dict(backbone='ResNet18',**x,role='REUSED_DEV_exploratory_unadjusted'))
 causal=[x for x in comparisons if x['contrast']=='P5_minus_P5_CONSTANT']
 assert len(causal)==4
 for x in causal:
  if x['metric']!='ALL_GT_PCK10': assert x['ci95_low']<0<x['ci95_high']
 return dict(schema='three_backbone_closeout_summary_v1',complete=True,report_only=True,
  new_fits=0,new_image_forwards=0,new_pose_solves=0,new_bootstrap_runs=0,scientific_goal_complete=False,
  stable_joint_TR_established=False,independent_test_present=False,accuracy_rows=accuracy,
  runtime_rows=runtime,runtime_overhead=overhead,direct_corner8_rows=direct,direct_pose_rows=direct_pose,
  paired_rows=comparisons,dimension_support=d['direct_FULL']['dimension_support']['real'],
  pose_contracts=dict(main='canonical_MAIN_R_cf_vs_R_gt_representative; W/D parity reported separately as axis_accuracy; GT extents',
   direct='physical-frame_R_cf@Q_vs_physical_GT; selected predicted extents',
   auc='proper-group_C2_aware_corresponding-point_ADD_AUC; not unrestricted nearest-neighbor ADD-S',
   FULL_axis_correct_frames=round(d['resnet']['methods']['FULL']['pose']['axis_accuracy']*319),
   FULL_axis_incorrect_frames=319-round(d['resnet']['methods']['FULL']['pose']['axis_accuracy']*319),
   direct_FULL_AUC=d['direct_pose_FULL']['summary']['ADDsym_AUC_full'],
   canonical_FULL_AUC=d['resnet']['methods']['FULL']['pose']['add_sym_auc_full_population'],
   direct_rotation_not_used_for_refiner_delta=True),
  runtime_scope={k:rt[k] for k in ['frames','sessions','repeats','batch','warmup_calls','measured_calls','maximum_2d_parity_abs_px','maximum_pose_parity_abs','limitations']},
  caveats=['Accuracy means per-seed statistics; latency is seed1.',
   'Conditional supports differ between backbones; no cross-backbone accuracy ranking.',
   'Direct corner8/pose endpoint is separate from the canonical correction evaluation.',
   'P5 minus P5_CONSTANT isolates head dimension input; P5 minus P0 does not.',
   'Square dimensions are fixed per population; no within-population dimension-effect identification.'])

def csv_export(name,rows):
 with (DOC/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
  for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in row.items()})
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def rel(alias):return '../'+SOURCES[alias]
def render(s):
 a=s['accuracy_rows'];main=[x for x in a if x['main_comparison']]
 accuracy=table(['기반','방법','2D median px↓','P90 px↓','ALL-GT PCK10 %↑','matched / 319','T cm↓','R °↓','pose coverage %'],
  [[r['backbone'],r['method'],f"{r['median_px']:.3f}",f"{r['p90_px']:.3f}",f"{100*r['all_gt_pck10']:.2f}",f"{r['matched_frames']:.0f}/319",f"{r['translation_median_cm']:.3f}",f"{r['rotation_median_deg']:.3f}",f"{100*r['pose_coverage']:.2f}"] for r in a])
 latency=table(['기반 / 대표 head','baseline 2D ms','보정 2D ms','paired 추가 2D ms','baseline 전체 ms','보정 전체 ms','paired 추가 전체 ms'],
 [[r['backbone']+' / '+r['correction'],*[f'{r[k]:.3f}' for k in ['baseline_two_d_median_ms','correction_two_d_median_ms','paired_added_two_d_median_ms','baseline_full_median_ms','correction_full_median_ms','paired_added_full_median_ms']]] for r in s['runtime_overhead']])
 marginal=table(['보조 통계: 각 arm median 차이','Δ2D ms','Δ전체 ms'],[[r['backbone'],f"{r['delta_two_d_median_ms']:.3f}",f"{r['delta_full_median_ms']:.3f}"] for r in s['runtime_overhead']])
 paired=table(['비교','지표','차이 [세션 95% CI]'],[[r['backbone']+' '+r['contrast'],r['metric'],f"{r['delta']:+.6f} [{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]"] for r in s['paired_rows']])
 direct=table(['집단','arm','matched / 전체','corner8 median px','P90 px','ALL-GT PCK10 %'],[[r['population'],r['arm'],f"{r['matched_frames']}/{r['frames']}",f"{r['median_corner8_px']:.3f}",f"{r['p90_corner8_px']:.3f}",f"{100*r['all_gt_pck10']:.2f}"] for r in s['direct_corner8_rows']])
 pose=table(['direct arm','T median cm','R median °','PnP available'],[[r['arm'],f"{r['translation_median_cm']:.3f}",f"{r['rotation_median_deg']:.3f}",f"{r['available']}/{r['frames']}"] for r in s['direct_pose_rows']])
 def m(backbone,method):return next(x for x in a if x['backbone']==backbone and x['method']==method)
 yr,yp=m('YOLO','R0'),m('YOLO','P');dr,dp=m('DOPE','DOPE'),m('DOPE','P');rr,rp=m('ResNet18','FULL'),m('ResNet18','P0')
 return f'''# 세 백본 보정 실험 최종 정리

기존 봉인 결과에서 YOLO·DOPE·ResNet18 각각의 보정 전후 조건부 2D 중앙값 감소를 확인했다. **치수 입력 자체의 안정적인 T·R 공동 개선이나 독립 TEST 일반화는 입증되지 않았다.** 이 문서는 결과 정리의 완료를 뜻하며 연구 목표 전체의 달성이나 투고·채택 완료를 뜻하지 않는다. 새 학습·추론·PnP·bootstrap 실행은 모두 0회다.

## 논문에서 방어할 수 있는 세 가지 결과

1. 재사용 DEV319에서 backbone별로 따로 학습한 local probability 보정은 조건부 2D 중앙값을 YOLO **{yr['median_px']:.3f}→{yp['median_px']:.3f}px**, DOPE **{dr['median_px']:.3f}→{dp['median_px']:.3f}px**, ResNet18 FULL+P0 **{rr['median_px']:.3f}→{rp['median_px']:.3f}px**로 낮췄다. 세 비교의 탐색적 세션 구간도 감소 방향이다. 하나의 공통 head 가중치를 다른 backbone에 무학습 전이한 실험은 아니다.
2. 동일 RTX 3080·동일 26영상에서 대표 seed1 head의 전처리부터 2D 출력 및 같은 MAIN PnP까지 다시 측정했다. 보정은 추가 latency를 요구한다. 아래 비용은 측정 범위를 고정한 데스크톱 기술 통계이며 Jetson·처리량·새 정확도 반복이 아니다.
3. 치수 조건부 ResNet과 P5의 통제 비교는 효과의 한계를 드러낸다. P5−P5_CONSTANT의 2D·T·R 구간은 모두 0을 포함한다. 치수의 추가 정보가 세 지표를 함께 안정적으로 개선한다는 결론은 낼 수 없다.

## Backbone 안에서의 보정 전후

![동일 backbone 내 보정 전후](figures/within_backbone_accuracy.png)

{accuracy}

주 비교는 YOLO P, DOPE P, ResNet P0다. ResNet P0는 동일 local probability 보정 원리를 잇는 치수 없는 head이며, 통합 runtime 주 비교도 P0_S1을 사용했다. 더 낮은 DEV 수치를 보고 P5로 주 비교를 교체하지 않았다. D/D0와 P5/P5_CONSTANT는 함께 공개한 통제·보조 비교다.

보정 수치는 seed1/2/3에서 계산한 통계의 산술평균이며 예측 앙상블이 아니다. 세 기반은 같은 DEV319·13세션·감독 landmark 2,818점을 사용하지만 조건부 매칭은 **YOLO 311장/2,756점, DOPE 190장/1,710점, ResNet 299장/2,644점**이다. median/P90은 고정 baseline box IoU≥0.5 및 9점 유한 조건에 따른다. ALL-GT PCK10은 제외·결측점을 실패로 포함한다. DOPE pose는 210/319만 유효하고 나머지 109장의 실패를 coverage에 남긴다. 따라서 조건부 숫자만으로 세 backbone의 정확도 순위를 정하지 않는다.

DOPE P와 ResNet P0의 P90은 baseline보다 약간 커졌다. 작은 중앙값 감소는 모든 프레임·꼬리 오차의 개선을 뜻하지 않는다. YOLO P도 기존 PoseFix-derived PRIOR보다 중앙오차가 0.336232px 높았으며 최상 정확도 주장이 아니다.

[YOLO 원문]({rel('yolo_report')}) · [YOLO 수치]({rel('yolo')}) · [DOPE 원문]({rel('dope_report')}) · [DOPE 수치]({rel('dope')}) · [ResNet 보정 V3]({rel('resnet_report')}) · [ResNet 수치]({rel('resnet')})

## 동일 조건에서 재측정한 실행시간

![동일 장치와 패널의 runtime](figures/runtime_overhead.png)

{latency}

주 추가 시간은 동일 frame ID·repeat의 보정 시간−baseline 시간을 먼저 계산한 **130개 paired 차이의 median**이다. 봉인 runtime의 `paired_added_two_d_ms`와 `paired_added_full_ms`를 인용했고 전체 측정 행에서도 다시 검산했다. 각 arm의 median 차이와는 다른 통계이며, 아래에 보조 값으로 함께 보존한다.

{marginal}

예를 들어 YOLO paired 추가 전체 시간은3.419ms이고 각 arm median 차이는3.687ms다. 전체 median은 2D median과 PnP median의 합으로 만들지 않고 저장된 전체 시간에서 직접 가져왔다. 정확도는 3-seed 평균, latency는 seed1이다.

batch1, 이미 RAM에 디코딩한 native BGR 시작, arm당 warmup20회와 26장×5반복=130회다. 주 6arm과 ResNet 보조 3arm을 합쳐 1,170 측정 호출을 모두 보존했다. 파일 읽기·디코딩·모델 로드·검산은 timer 밖이다. CPU Torch intra/inter-op 및 OpenCV thread는 각각1이며 GPU 수치 설정은 각 원 실험과 맞췄다. DOPE의 검출/PnP 실패 호출도 시간에 포함되므로 성공 경로만의 비용과 다르다. 단일 시점 관측이며 CI나 에너지·실시간 보장은 없다.

기존 개별 보고서의 오래된 latency는 이 표와 섞지 않았다. ResNet V3의 “해당 실행에는 latency 없음”은 당시 정확도 실행의 설명이고, 여기서는 나중에 완료한 별도의 통합 runtime을 연결한다. [통합 속도 원문]({rel('runtime_report')}) · [고정 protocol]({rel('runtime_protocol')}) · [1,170행 원결과]({rel('runtime')}) · [9arm CSV](RUNTIME_SUMMARY.csv)

## 치수 입력은 어디에 들어갔는가

| 경로 | 추정기 신경망 | 보정 head | PnP |
|---|---|---|---|
| 기존 YOLO + P | RGB | 이미지 특징·기존 점/box, 치수 입력 없음 | K·등록 물리 치수 |
| DOPE + P | RGB | 이미지 특징·기존 점/box, 치수 입력 없음 | K·등록 물리 치수 |
| ResNet FULL + P0 | RGB + 정규화 W,D,H·비율 5값 | 치수 벡터 없음 | K·등록 물리 치수 |
| ResNet FULL + P5 | RGB + 치수 5값 | 치수 5값 추가 | K·등록 물리 치수 |
| ResNet FULL + P5_CONSTANT | 동일 FULL | 동일 P5 구조·초기값, 문맥만 0 | K·등록 물리 치수 |

보정기 증분 치수 효과는 **P5−P5_CONSTANT**로 판단한다. P5−P0는 경로·용량도 달라지는 package 비교이고 P5−FULL은 보정 전체의 효과다. 모든 ResNet head는 이미 치수 조건부 FULL을 공유하므로 시스템 전체의 RGB-only 대조군은 아니다.

{paired}

2D/T/R 차이는 음수가 개선이고 PCK 차이는 양수가 개선이다. 표의 PCK 차이는 0–1 비율 단위이며 100을 곱하면 percentage point다. 13세션 bootstrap10,000회, seed20260914의 기존 결과를 그대로 옮겼으며 다중비교 보정이 없다. P5−P5_CONSTANT의 PCK10 구간만 양수인 사실로 2D/T/R 전체의 인과 성공을 선언하지 않는다. ResNet P0−FULL의 T 구간도 0을 포함한다. [claim audit]({rel('claim_audit')}) · [전체 paired CSV](PAIRED_SUMMARY.csv)

## Direct ResNet 치수 실험은 별도 endpoint

CONSTANT/SHAPE/FULL 세 arm은 같은 구조·학습 순서의 고정 epoch10, 각 34,990 updates·559,800 source exposures이며 arm당 학습 seed는 하나다. FULL은 canonical W,D,H와 두 비율, SHAPE는 두 비율, CONSTANT는 0 문맥이다. 실패한 초기 heatmap-MSE 모델과 이후 진단을 덮어쓴 결과가 아니라 별도 DSNT 학습의 완료 결과를 인용한다.

{direct}

이 표는 **관측 corner8·whole-object symmetry 평가**다. 위 보정 표의 canonical 9점 및 pose 평가와 분모·endpoint가 달라 FULL 수치가 같지 않다. 예를 들어 direct FULL DEV319 median은 8.012px이고 보정용 FULL은 8.001px다. 두 값이나 서로 다른 rotation 값으로 보정 효과를 계산하지 않았다. Direct pose는 `R_physical=R_cf@Q`를 physical-frame GT와 비교한다. 주 세 백본·보정기 표의 canonical MAIN은 `R_cf`를 `R_gt_representative`와 비교하며 W/D parity는 `axis_accuracy`로 따로 보고한다. FULL의 physical-frame R은3.581°·AUC0.316284이고 canonical R은3.252°·AUC0.312959다. Translation은 수치 정밀도 범위에서 같고, 축이 맞는234/319프레임은 회전 계약이 일치하지만 축 오류85프레임은 달라진다. Direct IoU도 선택된 예측 extents를, canonical 평가는 GT extents를 사용하므로 축 오류에서 차이가 난다. 차이가 R에만 한정된다고 해석하지 않는다. 여기의 AUC는 허용된 proper-group C2 변환에서 대응점 ADD를 계산한 값이며, 최근접점에 자유롭게 대응하는 ADD-S가 아니다. 같은 direct physical-frame 평가의 pose는 아래와 같다.

{pose}

Direct FULL은 CONSTANT보다 DEV T median이 악화되고 R median은 감소했다. SHAPE보다도 T가 크다. 치수 입력이 pose 두 축을 함께 개선했다고 볼 수 없다. ResNet은 한 팔레트가 있다고 가정해 9점을 항상 출력하므로 direct matched coverage는 detector recall이 아니고 pose coverage는 PnP 해의 존재 비율이다. [direct 보고서]({rel('direct_report')}) · [직접 실험 CSV](DIRECT_DIMENSION_SUMMARY.csv)

플라스틱 canonical **W×D×H=110×130×11cm**, 목재 **80×59×14cm**, 정사각형 **110×110×15cm**다. long/short 정렬 순서와 canonical 축 순서를 섞지 않는다. source TRAIN의 W/D/H 축별 범위는 각각0.590–1.363 / 0.818–1.720 / 0.064–0.244m다. 목재 D=0.59m는 밖에 있고 정규화 문맥도 일부 범위를 벗어나므로 외삽이다. 범위 안인 경우도 분포 동일성을 보장하지 않는다.

GREEN150과 0918-119는 각 집단에서 치수가 모두 고정돼 있어 프레임별 치수 변화의 인과 효과를 식별할 수 없다. 0918은 GREEN150과 encoded/decoded RGB가 중복되지 않지만 독립 TEST는 아니며 `split=train`, `population_role=DEV`가 함께 기록돼 있다. 119장에 manual click602점(이미지 안600점), PnP 생성350점, 자동 center119점이 있다. canonical pose·signed axis가 확정되지 않아 T/R 정답 집단으로 쓰지 않았다. 정사각형 결과는 direct 모델에 한정하며 이번 ResNet correction head는 두 square 집단에서 평가하지 않았다. [0918 자료 감사]({rel('square_report')})

## 기존 실제 RGB와 물리 치수

아래는 이미 생성된 원본 갤러리를 상대경로로 임베드했다. 새 이미지 생성·추론은 하지 않았다. DOPE와 direct ResNet의 개선·악화 극단 사례로서 결과를 보고 선택된 설명용 그림이며 집계 성능을 대표하지 않는다. direct 정사각형 그림을 P0/P5 보정 결과로 해석하면 안 된다.

![DOPE 개선 사례](../{IMAGES[0]})

![DOPE 악화 사례](../{IMAGES[1]})

[DOPE 전체 RGB·K·치수 및 선택 manifest]({rel('dope_gallery')})

![0918 direct ResNet 개선 사례: 110×110×15cm](../{IMAGES[2]})

![0918 direct ResNet 악화 사례: 110×110×15cm](../{IMAGES[3]})

[direct 전체 갤러리]({rel('direct_gallery_md')}) · [ID·원영상·치수·표시 선택 manifest]({rel('direct_gallery')})

## 원고용 문장과 제한

사용 가능한 표현: “동결된 세 기반 추정기에 각각 source-only로 학습한 local probability 보정기를 적용했을 때, 재사용 DEV319에서 기반별 조건부 2D 중앙오차가 감소했다. 같은 영상·장치에서 대표 head의 추가 지연시간도 측정했다.” 이 결과는 세 고정 설정에 대한 적용 근거이며 모든 backbone에 대한 무학습 일반화 증명이 아니다.

치수 관련 표현: “ResNet baseline과 보정 head의 치수 입력을 통제 비교했다. P5의 상수 문맥 대조군 대비 증분은 작고 2D·T·R의 탐색적 신뢰구간이 0을 포함했으며, 안정적 공동 개선은 확인되지 않았다.”

실사 DEV는 반복 개발에 사용됐고 독립 TEST가 없다. Pose reference는 2D 주석·K·등록 치수로 재구성했으며 독립 물리 계측 6D GT가 아니다. 새 보정 학습은 합성 source만 사용했지만 전체 시스템의 모든 사전학습·교사 이력까지 real-GT-free라고 주장하지 않는다. Backbone 초기화·훈련 예산·usable source support가 다르므로 백본 자체의 우열이나 동일 학습 예산 실험으로 주장하지 않는다. 새로 넣은 학습 치수 기능은 ResNet과 P5에 해당하며 YOLO/DOPE 기존 P에도 적용됐다고 소급하지 않는다.

## 출처와 재검증

[SOURCE_BINDINGS.json](SOURCE_BINDINGS.json)은 원 JSON/MD/기존 그림의 정확한 SHA-256·크기를 고정한다. [REPORT_MANIFEST.json](REPORT_MANIFEST.json)은 검증한 기존 receipt 연결, 출력 hash, 그림 입력값과 실행 범위를 기록한다. [SUMMARY.json](SUMMARY.json), [정확도 CSV](ACCURACY_SUMMARY.csv), [속도 CSV](RUNTIME_SUMMARY.csv), [속도 차이 CSV](RUNTIME_OVERHEAD.csv), [paired CSV](PAIRED_SUMMARY.csv), [direct CSV](DIRECT_DIMENSION_SUMMARY.csv), [direct pose CSV](DIRECT_POSE_SUMMARY.csv)가 전체 정밀도를 보존한다.

[생성 코드](../../../scripts/research/{HERE.name}/report.py) · [검증 테스트](../../../scripts/research/{HERE.name}/test_report.py) · [테스트 결과](TEST_RESULTS.json)

기존 metric을 원영상/GT에서 다시 평가하지 않았다. 검증 범위는 봉인 JSON/MD의 byte 일치, 기존 receipt 연결, seed 통계와 runtime1,170행의 재집계, 표·그림·CSV·링크 일치다. 원 checkpoint·대용량 raw 데이터는 열거나 복사하지 않았다. 기존 파일과 원고를 수정하지 않았고 git add/commit/push 또는 학술지 제출을 수행하지 않았다.
'''

def figures(s):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 out=DOC/'figures';out.mkdir(exist_ok=True)
 names=['YOLO','DOPE','ResNet18'];pairs=[]
 for name in names:
  pairs.append([r for r in s['accuracy_rows'] if r['backbone']==name and r['main_comparison']])
 fig,axs=plt.subplots(1,2,figsize=(11.8,4.8),layout='constrained')
 xs=np.arange(3);width=.33;colors=['#6c7d8e','#117a75']
 for ax,key,title in zip(axs,['median_px','all_gt_pck10'],['Conditional 2D median (px, lower better)','All-GT PCK10 (%, higher better)']):
  for idx,label in enumerate(['Frozen baseline','Local correction (3-seed mean)']):
   vals=[p[idx][key]*(100 if key=='all_gt_pck10' else 1) for p in pairs]
   bars=ax.bar(xs+(idx-.5)*width,vals,width,color=colors[idx],label=label)
   ax.bar_label(bars,fmt='%.2f',padding=4,fontsize=10)
  ax.set_xticks(xs,['YOLO + P\n311/319 matched','DOPE + P\n190/319 matched','ResNet18 + P0\n299/319 matched'])
  ax.set_title(title);ax.set_ylim(0,ax.get_ylim()[1]*1.15);ax.grid(axis='y',alpha=.18)
 axs[0].legend(loc='upper left',fontsize=8)
 fig.suptitle('Within-backbone changes on reused DEV319\nDifferent conditional support; separate heads; no ensemble',fontsize=13)
 for ext in ['png','pdf']: fig.savefig(out/f'within_backbone_accuracy.{ext}',dpi=190,metadata={'CreationDate':None} if ext=='pdf' else None)
 plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(11.8,4.8),layout='constrained')
 rows=s['runtime_overhead']
 for idx,(key,label) in enumerate([('baseline_full_median_ms','Baseline'),('correction_full_median_ms','Correction, seed 1')]):
  bars=axs[0].bar(xs+(idx-.5)*width,[r[key] for r in rows],width,label=label,color=colors[idx]);axs[0].bar_label(bars,fmt='%.2f',padding=4)
 for idx,(key,label) in enumerate([('paired_added_two_d_median_ms','Paired added 2D median'),('paired_added_full_median_ms','Paired added full median')]):
  bars=axs[1].bar(xs+(idx-.5)*width,[r[key] for r in rows],width,label=label,color=['#bc8045','#8067a1'][idx]);axs[1].bar_label(bars,fmt='%.2f',padding=4)
 for ax in axs:
  ax.set_xticks(xs,['YOLO + P1','DOPE + P1','ResNet18 + P0_S1']);ax.set_ylabel('Milliseconds');ax.grid(axis='y',alpha=.18);ax.set_ylim(0,ax.get_ylim()[1]*1.22);ax.legend(fontsize=8,loc='upper left')
 axs[0].set_title('End-to-end median: decoded RGB through MAIN PnP')
 axs[1].set_title('Paired added latency (same frame and repeat)')
 fig.suptitle('Same RTX 3080 / 26 images / batch 1 / 5 repeats\nFailures retained; file decode and model loading excluded',fontsize=13)
 for ext in ['png','pdf']:fig.savefig(out/f'runtime_overhead.{ext}',dpi=190,metadata={'CreationDate':None} if ext=='pdf' else None)
 plt.close(fig)
 return dict(accuracy_pairs=pairs,runtime_overhead=rows)

def validate_links(text):
 checked=[]
 for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
  if target.startswith(('http:','https:','#')):continue
  p=(DOC/target.split('#')[0]).resolve();assert p.is_relative_to(ROOT) and p.exists(),target
  checked.append(str(p.relative_to(ROOT)))
 return checked

def build():
 d,pin,edges=load();s=summarize(d)
 write('SUMMARY.json',s)
 for name,key in [('ACCURACY_SUMMARY.csv','accuracy_rows'),('RUNTIME_SUMMARY.csv','runtime_rows'),('RUNTIME_OVERHEAD.csv','runtime_overhead'),('PAIRED_SUMMARY.csv','paired_rows'),('DIRECT_DIMENSION_SUMMARY.csv','direct_corner8_rows'),('DIRECT_POSE_SUMMARY.csv','direct_pose_rows')]:csv_export(name,s[key])
 fv=figures(s);write('REPORT_KO.md',render(s))
 write('README.md', '''# 세 백본 실험 마감 자료

[한국어 전체 보고서](REPORT_KO.md) · [출처·출력 manifest](REPORT_MANIFEST.json) · [검증 결과](TEST_RESULTS.json)

세 고정 backbone의 별도 보정 학습에서 재사용 DEV의 조건부 2D 중앙값 감소를 관찰했다. 치수 입력의 안정적 T·R 공동 개선과 독립 TEST 일반화는 입증되지 않았다. 정확도는 세 seed 통계 평균이며 통합 runtime은 대표 seed1이다.

[전체 JSON](SUMMARY.json) · [정확도 CSV](ACCURACY_SUMMARY.csv) · [runtime CSV](RUNTIME_SUMMARY.csv) · [runtime 차이 CSV](RUNTIME_OVERHEAD.csv) · [paired CSV](PAIRED_SUMMARY.csv) · [direct 치수 CSV](DIRECT_DIMENSION_SUMMARY.csv) · [direct pose CSV](DIRECT_POSE_SUMMARY.csv)

그림: [정확도 PNG](figures/within_backbone_accuracy.png) / [PDF](figures/within_backbone_accuracy.pdf), [runtime PNG](figures/runtime_overhead.png) / [PDF](figures/runtime_overhead.pdf). 기존 RGB 갤러리는 전체 보고서에 상대경로로 연결되어 있다.

저장소 루트에서 기존 공개 source artifact를 유지한 채 CPU로 재생성한다. Python에 NumPy, Matplotlib, Pillow가 필요하다.

```sh
python -B scripts/research/pallet_three_backbone_closeout_20261002_v1/report.py build
python -B scripts/research/pallet_three_backbone_closeout_20261002_v1/test_report.py
```

SOURCE_BINDINGS.json은 고정된 원본 hash다. source가 달라지면 중단하며 자동 재봉인하지 않는다. 원영상/GT 재평가·GPU·checkpoint·대용량 raw 캐시는 필요하지 않다. 기존 artifact 및 manuscript를 수정하지 않는다.
''')
 # No self-hash or future test receipt hash: verification is a separate final step.
 outputs=[p for p in DOC.rglob('*') if p.is_file() and p.name not in ['REPORT_MANIFEST.json','TEST_RESULTS.json']]
 manifest=dict(schema='three_backbone_closeout_manifest_v1',complete=True,report_only=True,
  source_pin=bind(DOC/'SOURCE_BINDINGS.json'),inputs=pin['inputs'],historical_receipt_edges=edges,
  codes=[bind(HERE/'report.py'),bind(HERE/'test_report.py')],outputs={str(p.relative_to(DOC)):bind(p) for p in sorted(outputs)},
  figure_values=fv,new_fits=0,new_image_forwards=0,new_pose_solves=0,new_bootstrap_runs=0,
  checkpoints_read_or_copied=0,raw_cache_read_or_copied=0,existing_files_modified=0,
  manuscript_modified=False,git_publication_performed=False,scientific_goal_complete=False,
  stable_joint_TR_established=False,independent_test_present=False,
  verification_scope='Public artifact bindings and fixed-result arithmetic; not a fresh GT/model evaluation.',
  same_generator_tests_are_not_independent_research_replication=True)
 write('REPORT_MANIFEST.json',manifest)
 print(json.dumps(dict(complete=True,report=bind(DOC/'REPORT_KO.md'),manifest=bind(DOC/'REPORT_MANIFEST.json'),rows={k:len(s[k]) for k in ['accuracy_rows','runtime_rows','paired_rows','direct_corner8_rows']})))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['seal','build']);args=parser.parse_args()
 seal() if args.phase=='seal' else build()
