from common import *
import shutil,zipfile
fit=read(ROOT/'FIT_COMPLETE.json');freeze=read(ROOT/'MODEL_FREEZE.json');check=read(ROOT/'INDEPENDENT_METRIC_CLOSURE.json');cls=list(csv.DictReader(open(ROOT/'CLASSIFICATION_METRICS.csv')));summ=[]
for scope in ['SYNTH','REAL']:
 for r in csv.DictReader(open(ROOT/f'{scope}_POSE_SUMMARY.csv')):summ.append(dict(scope=scope,**r))
csvsave('ALL_POSE_SUMMARY.csv',summ)
report='''# v60: 합성 external-ray proxy head의 실제 학습과 동결 평가

**결론: proxy 분류는 학습됐지만, 사전에 고정한 자세 채택 기준은 실패했다. 현재 구성은 채택 보류(HOLD)다.** 추가 학습이나 실사 결과를 이용한 설정 변경은 하지 않았다. 원 OCPR 또는 Pro의 physical DIRECT/위치/joint 4설정 실험을 완성한 결과가 아니다.

## 실제 수행 범위

- 1개 head, seed 20261018, 20 epoch, optimizer **480 update**, 28,033 parameter. 추가 fit 0회. 첫 실제 update는 2026-10-22 11:01:03 UTC다.
- 합성 TRAIN768/VAL235의 기존 분할을 사용했다. TRAIN query 5,690개(양성674), VAL1,749개(양성228). centroid는 감독에서 제외했다. 원래 유효 TRAIN query 중 1개는 9×9 grid의 원본 영상 픽셀 지지가 없어 제외됐다.
- 목표는 native 첫8 query의 `occlusion_fraction >= 0.5`이다. 이는 최대9개의 광선 중 외부 첫-hit 차단 비율이며 **물리 코너 DIRECT/EXTERNAL 정답이 아니다**. native 가상 cuboid 점을 실제 표면으로 옮기지 않았다.
- 기존 YOLO26 P3를 동결하고 BN running statistics를 그대로 보존했다. 9×9 ROI에 원본 pixel-valid 채널을 붙였다. 특징 통계는 실사에 맞춰 적합하지 않았다. 좌표 회귀나 joint head는 없다.
- threshold0.5, 최소 native6점, W/D 가설/score, SQPnP+LM, centroid score, 부족점 시 정확한 저장 N0 fallback을 고정했다. 예측 외부 ray 차단점만 제외하고 새로운 self gate는 넣지 않았다. 원좌표와 재투영점을 독립 관측으로 중복 투입하지 않았다.
- VAL BCE가 가장 낮은 epoch15를 선택하고 실사 평가 전에 체크포인트를 동결했다. 실사 이미지/특징/레이블로 학습·normalizer·calibration·threshold·checkpoint를 선택한 횟수는0이다. real-CAL state/covariance/reliability/prior/O/T/H1과 MobileSAM은 이 경로에서 사용하지 않았다.

## 분류 결과와 자세 결과는 다르다

VAL proxy BCE **0.12830** (TRAIN-prior 상수0.38773), AUROC **0.97020**, AP **0.86902** (VAL 양성 비율0.13036). threshold0.5에서 TP175/FP32/FN53/TN1489, precision0.84541, recall0.76754이다. 분류 screening은 통과했다. 이는 실제 가림 판별 정확도나 자세 성능 향상의 증거가 아니다.

공동 성공은 T<5cm, 등록된 대칭 최소 R<5도이다. 미산출도 전체 분모에 남긴다. 아래 모든 경우 fallback 이후 산출은 전량이다. 자세/좌표 출력은 참조를 읽기 전에 봉인했다.

| 자료 | N0 성공→head 성공 | 회복/손상 | 평균 T(cm) | 평균 R(도) | P95 T(cm) | P95 R(도) | T>50 | head fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAIN768 |294→290|11/15|15.556→15.910|38.913→39.174|57.088→62.399|179.603→179.621|47→48|51|
| VAL235 |52→50|3/5|14.312→14.308|94.899→94.306|66.879→66.677|179.869→179.861|16→16|25|
| 목재 PRIMARY105, C2 |57→59|3/1|15.679→15.740|12.867→12.887|84.251→84.379|85.633→85.633|8→8|1|
| 정사각형118, C4 |30→26|2/6|9.378→9.998|3.344→3.310|23.233→27.264|5.417→5.413|1→1|2|

mean/median/P90/P95/max, 산출/미산출, canonical 회전과 대칭 회전, 큰 오류는 `ALL_POSE_SUMMARY.csv`, 모든 프레임의 회복/손상은 `*_POSE_PAIRED.csv`에 있다. 목재 median T3.404→3.404, R2.328→2.342; 정사각형 median T7.170→7.760, R1.640→1.563이다. 실사 대칭 R>45는 목재13→13, 정사각형1→1이다. canonical 약90도(80–100도)는 목재12→11, 정사각형13→13이다. 이를 W/D·semantic 대응 오류의 확정 정답으로 해석하지 않는다.

사전 gate는 순회복>0, T>50 증가 없음, 평균 T/R 및 P95 T/R 악화 없음이다. VAL은 순회복에서 실패했다. 목재는 성공 수가 증가했지만 평균 T/R과 P95 T에서 실패했다. 정사각형은 순회복·평균 T·P95 T에서 실패했다. **평균 개선으로 성공 손상을 숨기거나, 성공 증가만으로 tail 악화를 숨기지 않는다.** 결과는 trade-off이며 기존 기준 통과가 아니다. 실사 평가 자체는 학습 전 명시한 동결 평가로 진행했으며, 부정 결과 이후 새 seed/threshold를 시도하지 않았다.

## 관찰 가능한 원인과 한계

- VAL 손상5장 모두 제외한 좌표의 저장 ray target은 양성이었다(해당 9개 제외점). 따라서 이 손상은 분류 오탐만으로 설명되지 않는다. **ray 차단 신호가 올바르다는 것과 좌표를 PnP에서 빼면 좋아진다는 것은 다른 목적**이다. 물리 표면 DIRECT 진실을 확보한 결과도 아니다.
- 2점 제외하여6점이 남는 경우에 성공 변화가 집중됐다. VAL 회복3/손상4가 그 층에 속하고, 나머지 손상1은7점이다. 정사각형의 회복2/손상6은 모두6점 층이었다. 단순 point count만으로 기하 안정성·대응 정확성이 보장되지 않는다. 새로운 최소점수나 threshold로 바꾸지 않았다.
- 목재에서 W/D family 변경은0/105, 정사각형은 동일 W/D가 합쳐져 family 변경0/118이다. 이번 실사 성공 변화는 큰 분기 교체가 확인된 결과가 아니다. 관측점을 일부 제거한 동일 가설 내 PnP 변화도 translation을 크게 바꿀 수 있다.
- 그림의 목재 회복004687은 T7.81→2.02cm, 손상026673은 T1.37→8.07cm다. 정사각형 회복029476은 T5.34→2.71cm, 손상028389는 T4.48→32.72cm다. 실제 이미지와 저장된 투영을 함께 확인했다. 제외점의 실제 물리 가림 정답은 실사에 없어 UNKNOWN이다.
- v57/v58의 고정8장 convex-self+external oracle 부정 결과와 physical support 미확인/가상점 한계는 보존됐다. v60은 다른 external-only proxy 정책이며 이를 그 oracle 또는 원 Pro DIRECT 실험의 통과로 바꾸지 않는다.

## 실사 입력 복구와 재현

첫 parity에서 목재 REMAINDER80장이 실패했다. 원인은 관측용 trace가 초기 self-hidden 좌표를(-1,-1)로 지운 carrier를 저장했는데 원 solver 좌표로 취급한 것이다. 원본 journal SHA041afb75...의 `qFinal == native_points`, `actual_pose == initial_pose`를 확인하여 **실제 저장된 원좌표**를 복원했다. 점을 새로 예측하거나 GT로 채우지 않았다.

또한 `Q` 역산으로 구한 치수의 끝자리 오차 대신 당시 등록 metadata의 [.8,.14,.59]m를 그대로 사용했다. `K`는 원 readiness metadata와 일치했다. tolerance1e-9를 완화하지 않았으며 최종223장 parity 최대 pose component 차이는8.81e-10이다. 첫 실패와 복구1의 수치 실패(9.23e-9)도 그대로 남겼다. 최종 입력은 `PREPARED_INPUTS_REAL_VERIFIED_V2.json.gz`, 검사는 `REAL_VERIFIED_PARITY_V2.json`이다. 복구는 실사 성능 채점 전에 끝났다.

## 학습 출처·평가 제한

기본 YOLO26 checkpoint970a0913...는 합성 팔레트 G38/P0/TEX 학습, N3 좌표는 기존 합성 학습 checkpointceea743b...의 저장 출력이라는 retained 계약을 사용했다. 범용 초기 가중치 전체가 합성으로만 만들어졌다고 주장하지 않는다. N3 새 forward0, SAM0, real-CAL fit0이다. `SOURCE_PROVENANCE.csv`는 기존 모듈별 출처 감사로, 과거 OCPR의 real-CAL 의존성과 이번 사용 제외를 구분한다.

합성 scene ID와 RGB hash가 TRAIN/VAL에 겹치지 않음을 확인했다. 하지만 전체 원본 paired-render/증강/배경 자산 family 계보를 독립 재구축하지 못했고, 기존 detector validation 노출 및 반복 VAL 개발 이력이 있다. 새로운 독립 synthetic test의 결과가 아니다. 목재PRIMARY와 정사각형118도 반복 노출 개발자료다. 실제 치수는 유형별 등록값이며 개별 실물 실측 검증은 제한된다. 실사 참조는 기하 재구성 참조이며 독립 물리 pose GT가 아니다. C2/C4는 기존 봉인 평가 계약을 유지했고 외형만으로 새 대칭을 부여하지 않았다.

역사적 원 OCPR seed18/PF 비교는 별도 `HISTORICAL_REFERENCE_SUMMARY.csv`로 보존한다. 목재 OCPR65/PF63, 정사각형 OCPR41/CPU-PF37 성공이라는 기존 행은 이번 head 결과가 아니다. 특히 원 OCPR은 real-CAL 학습/통계에 간접 의존한다. 새 head의 synthetic fit으로 그 과거 경로까지 synthetic-only로 바꾸지 않는다.

## 계산·검산·산출물

RTX3080 동일 Ubuntu 실행. SYNTH1003 frozen 특징+ROI/cache 저장19.855초, head480 update+epoch별VAL 평가2.292초, 합성 공통 후단+채점2.456초. REAL223 frozen 특징+ROI/cache4.247초, 동결 head forward0.101초, 공통 후단+채점0.332초. 모델 초기화 내부 forward를 포함하면 SYNTH1004/REAL224 backbone 호출(명시적1003/223)이다. GPU 설치/재부팅/다른 작업 종료는 하지 않았다.

이 시간은 **저장 N0 좌표/자세를 읽은 이후의 측정 구간**이며 Python 시작·모델 로드·N3/SubPix 생성·전체 RGB→pose end-to-end 시간은 아니다. 전체 속도 개선이라고 주장하지 않는다. 같은 데이터의 새 real training은0이다.

2452개 pose metric 행을 별도 scalar dot/acos·fsum 및 Python 통계로 재검산했다. 동일 metric 최대 차이6.67e-11이며 성공·회복·손상 수는 모두 일치한다. 초기 SciPy SO3 독립 검사는 입력 회전행렬을 SO3로 정규화하는 다른 metric이어서 최대.015881도 차이를 보였다. 최악 참조 R의 직교 오차1.19e-7이었고, 이 차이와 첫 실패는 `INDEPENDENT_*`에 공개 기록했다. 기존 trace 평가를 사후 바꾸지 않았다.

`Real_Proxy_Cases.pdf/png/svg`는 회복/손상/성공대조를 유형별1개씩 총6행 보여준다. 같은 원RGB/crop/scale, reference 흰 점선·pose 주황 실선·원좌표 청록 원·제외점 자홍×로 구분한다. 확대 점의 오차가 아닌 전체 pose T/R다. image/pose/K/치수/좌표/행 연결은 figure manifest에 있다. 패널을 실제 열어 검수했다. 사례는 사후 목적별 lexical recording/ID 선택이며 전체 우위의 증거가 아니다.

원 저장소 tracked6개 dirty 파일, 별도 scientific checkout tracked clean 상태, 원고/기존 실험을 보존했다. 외부 업로드·공개·원고 변경은0회다. 학습/평가 프로세스는 완료됐다. 현재 결과를 개선 달성으로 선언하거나 추가 sweep을 자동 시작하지 않는다.
'''
(ROOT/'REPORT_KO.md').write_text(report)
(ROOT/'REPRODUCE.txt').write_text('''Scope: local only; original source/data/checkpoints required and intentionally not bundled. Use a NEW output directory, not overwrite existing immutable receipts.
Scientific source revision:93c24d143fa3901de1fa127b3156e8250d81dba8
CPU env:/home/minjae/anaconda3/envs/pallet-pose/bin/python
GPU env:/home/minjae/anaconda3/envs/pallet-yolo26/bin/python
Threads:OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Order actually executed:
prepare_inputs.py
extract_features.py --scope SYNTH [same-host GPU]
fit_head.py [same-host GPU; exactly one fit480updates; no automatic rerun]
evaluate_pose.py --scope SYNTH [CPU; predictions sealed before references]
parity_real_inputs.py [fails80; preserve]
repair_real_carrier_attempt1.py [original qFinal restores80; inverse-dimension rounding still fails strict1e-9]
repair_real_carrier.py [exact registry dimensions; REAL_VERIFIED_PARITY_V2 PASS]
extract_features.py --scope REAL [same-host GPU; frozen model]
predict_real_head.py [same-host GPU; no optimizer/targets]
evaluate_pose.py --scope REAL [CPU; predictions sealed before references]
verify_and_figures.py [same-formula success counts; SciPy projected rotation diagnostic reported]
independent_metric_closure.py [same historical trace metric independent scalar audit]
finalize_local.py [reports/local package only]
No threshold/anchor/crop/checkpoint changes based on real performance.
See common.py absolute local source paths; files are private local deliverables, not public portable data release.
''')
shutil.copyfile(ROOT.parent/'synthetic_head_minimal_20261022_v54/G0/MODULE_PROVENANCE_AND_USE.csv',ROOT/'SOURCE_PROVENANCE.csv');shutil.copyfile(V59/'HISTORICAL/SUMMARY.csv',ROOT/'HISTORICAL_REFERENCE_SUMMARY.csv')
sources=[SRC/'scripts/research/pallet_line_pose_v1/features.py',SRC/'scripts/research/pallet_dim_conditioned_p_v1/pose.py',SRC/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',SRC/'challenge/evaluation_v2/pnp_selector.py',V58/'masked_backend.py',B/'pallet-boundary-action-repair-20261018-v1/d1_upstream_v1/SYNTHETIC1003_UPSTREAM.jsonl.gz',B/'pallet-statewise-end-to-end-recovery-v1/synthetic_v1/DATA_SPLIT_LOCK.json']
save('SOURCE_BINDINGS.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))for p in sources]);save('FINAL_STATUS.json',dict(UTC=utc(),status='COMPLETE_HOLD_NO_ADOPTION',training_fits=1,optimizer_updates=480,physical_DIRECT_training=0,location_or_joint_training=0,real_training_or_fit=0,synthetic_inference_frames=1003,real_evaluation_frames=223,wood=105,square=118,gate_all_groups=False,upload_attempts=0,manuscript_changes=0,ongoing_experiment_process=False,head_sha256=freeze['checkpoint_sha256'],independent_metric_PASS=check['PASS'],physical_support_blocker='Original Pro physical DIRECT supervision remains unestablished for unchanged native virtual corners; proxy feasibility does not solve it.'))
files=[p for p in ROOT.rglob('*')if p.is_file()and'__pycache__'not in str(p)and p.suffix not in ['.zip']and p.name!='MANIFEST_SHA256.csv'];csvsave('MANIFEST_SHA256.csv',[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))for p in sorted(files)])
selected=[p for p in ROOT.iterdir()if p.is_file()and p.suffix in ['.py','.md','.txt','.csv','.json','.pdf','.png','.svg','.pt']and'PREPARED_INPUTS'not in p.name];zipfile_path=ROOT/'Pallet_v60_Proxy_Feasibility_HOLD_Local.zip'
with zipfile.ZipFile(zipfile_path,'x',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
 for p in sorted(selected):z.write(p,p.name)
save('LOCAL_BUNDLE_RECEIPT.json',dict(path=str(zipfile_path),bytes=zipfile_path.stat().st_size,sha256=sha(zipfile_path),uploaded=False,includes_raw_RGB=False,includes_ROI_cache=False,includes_private_inputs=False))
print('LOCAL_FINAL_COMPLETE',zipfile_path.stat().st_size,flush=True)
