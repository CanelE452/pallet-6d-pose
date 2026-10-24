"""Close only owned V222 saved SOURCE evidence; never fit or read protected payloads."""
import os, stat, json, hashlib, shutil
from pathlib import Path
from datetime import datetime, timezone
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'publication_root_preparation'
S=P/'query_visibility_expected_part_counterfactual_source_bridge_20261025_v222'
def checked(p):
    p=Path(p)
    if 'v112' in str(p).lower(): raise RuntimeError('forbidden lexical path')
    return p
def raw(p):
    p=checked(p)
    if p.is_symlink(): raise RuntimeError('symlink')
    return p.read_bytes()
def bind(p):
    b=raw(p); return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}
def new(p,b):
    p=checked(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(b)
def jnew(p,o):new(p,(json.dumps(o,ensure_ascii=False,indent=2)+'\n').encode())
summary=json.loads(raw(S/'saved_source_comparison_v2/SOURCE_COMPARISON.json'))
fit=json.loads(raw(S/'paired_source_fit_v1/COMPLETE.json'))
modelqa=R/'V222_ACTUAL_MODEL_QA_A/actual_review_v1/REVIEW.json'
assert json.loads(raw(modelqa))['PASS'] is True
report='''# V222: 예상 부위 대조 감독의 SOURCE 검증 — 일반화 개선은 확인되지 않음

같은 C65 모델에서 시작한 두 모델을 각 1,100 update 실제 학습했다. C66은 기존 공개 실사·합성 SOURCE recipe를 계속 학습했고, C67은 여기에 **같은 query 조건에서 예상 부위를 가리는 대조 감독**을 더했다. C67은 새 학습 39묶음을 모두 맞혔지만, 검증 9묶음의 전체 정답은 C66보다 줄었다. 두 모델 모두 채택하지 않았다. 현재 평가 223장에 대한 새 추론은 하지 않았으며, 최신 실사 비교는 V220이다.

## 데이터와 정답의 의미

기존 공개 실사 SOURCE의 관찰된 부위 D48을 사용해 BEFORE(보임), COVER(해당 부위를 덮음), SHAM(같은 가림막을 해당 부위 밖으로 이동)을 만들었다. 48묶음은 nominal 19개와 jitter 29개이며 독립 장면 48개가 아니다. FIT 39묶음/117행, DEV 9묶음/27행이다. DEV는 2 frame/component의 4 nominal 부위와 그 jitter로 작고 상관돼 있다. 기존 natural weak는 FIT45(D39/E6), DEV12(D9/E3), strong point는 TRAIN6096(D5853/E243)이다. 서로 겹치는 BEFORE와 natural 행을 합산해 성능을 부풀리지 않았다.

새 대조 데이터의 라벨은 **관찰한 유한 부위 영역에 대한 약한 감독**이다. 물리적인 native 키포인트 한 점의 visibility 정답이 아니며 `native_point_DE_target=-1`을 유지했다. 이 공개 학습 family의 부모 q8은 GT에서 유래했다. 생산 단계의 초기 추론 query에 그대로 정렬됐다는 증거가 아니며, 별도 original initializer-query 9개를 이번 FIT에 사용하지 않았다. 부모 crop에 추가 GT 재중앙 정렬은 하지 않았다. 카드의 “No GT recrop”은 이번 producer에서 추가 recrop이 없다는 뜻이다.

세 상태의 q/L/native/q8/condition32/valid를 동일하게 유지했다. COVER는 두 관찰자의 weak ROI union을 완전히 덮고, SHAM은 같은 RGB/alpha mass를 valid 안의 target-disjoint 위치로 옮겼다. 모든 48묶음 생성에 성공했고, 실패 사례를 사후 제외한 것은 없다. 가림막은 seeded procedural stripe 1 family로 다양성이 좁다. SHAM은 다른 목재를 가릴 수 있으므로 모든 팔레트 부위나 실제 native 점을 보존한다는 뜻은 아니다. 청록 ROI와 빨간 중심 표시는 시각화에만 있으며 학습 RGB에는 없다.

[전체 데이터 표](expected_part_counterfactual_pairs_v1/ALL144_VARIANT_ROWS.csv) · [48묶음 생성/픽셀 증거](expected_part_counterfactual_pairs_v1/SUMMARY.json) · [전체 생성 갤러리](expected_part_counterfactual_pairs_v1/PORTABLE_GALLERY_KO.md)

## 실제 학습과 비교 조건

두 모델의 초기 tensor state는 같은 V219 C65이고 optimizer는 각각 비워서 시작했다. C65 선택은 완료된 SOURCE recipe의 연속성에 따른 것이며 평가셋 점수로 선택하지 않았다. 모델은 각 118,359 parameter이며 모두 학습했다. 공통 update마다 기존 SOURCE192 + strong point32 + natural weak16을 같은 RNG/draw로 사용했다. C67만 FIT 부모 4개×3상태를 추가했다. 기존 backbone·구조·normalizer·문턱을 바꾸지 않았다. 새 DINO/hinge 처방이 아니다.

C67의 추가 손실은 `0.25 * mean_parent(0.25*BCE(BEFORE,D)+0.5*BCE(COVER,E)+0.25*BCE(SHAM,D))`이다. 공통 240개와 추가 12개이므로 같은 example 수의 대조가 아니다. 추가 데이터와 손실 branch를 함께 바꾼 비교이며, 부위 주소만의 인과 효과를 분리한 실험이라고 할 수 없다.

먼저 각 32 update(합계64)의 throughput probe를 실행하고 모두 버렸다. 점수와 무관한 사전 예산식으로 final 1,100 update씩을 정한 뒤 두 모델을 다시 같은 C65에서 시작했다. final 합계2,200 update이며 학습 재실행은 없었다. 두 final checkpoint를 모두 동결한 뒤 DEV 출력을 만들었다. DEV RGB는 loss/sampling pool/normalizer에 들어가지 않았다. SOURCE DEV는 과거 관찰된 개발 자료라 새 blind test는 아니다.

[고정 recipe](ROOT_FIXED_SOURCE_PAIR_RECIPE.json) · [실제 학습 완료](paired_source_fit_v1/COMPLETE.json) · [두 모델 선동결](paired_source_fit_v1/ALL2_FINAL_FROZEN_BEFORE_ANY_DEV_OR_REAL_OUTPUT.json)

## 저장 출력 결과

D=해당 weak 부위 보임, E=해당 weak 부위 가림이다. 아래 CM은 행이 정답 D/E, 열이 예측 D/E/U다. z>0을 E로 하는 고정 이진 판정이며 문턱 조정은 없었다. U율0%는 불확실성을 해결했다는 뜻이 아니라 이 판정이 보류하지 않기 때문이다.

| 분리한 대상 | C66 기존 감독 계속 | C67 부위 대조 감독 추가 |
|---|---:|---:|
| CF FIT 39묶음 세 상태 모두 정답 | 1/39 | 39/39 |
| CF FIT117 CM | [[75,3,0],[36,3,0]] | [[78,0,0],[0,39,0]] |
| CF DEV9묶음 세 상태 모두 정답 | 1/9 | 2/9 |
| CF DEV27 CM | [[17,1,0],[7,2,0]] | [[14,4,0],[6,3,0]] |
| CF DEV 전체 정답 | 19/27 | 17/27 |
| CF DEV D precision / recall | 70.83% / 94.44% | 70.00% / 77.78% |
| CF DEV E precision / recall | 66.67% / 22.22% | 42.86% / 33.33% |
| CF DEV BEFORE D / COVER E / SHAM D 정답 | 9/9 · 2/9 · 8/9 | 7/9 · 3/9 · 7/9 |
| natural weak DEV12 CM | [[9,0,0],[1,2,0]] | [[7,2,0],[1,2,0]] |
| natural DEV D precision / recall | 90.00% / 100.00% | 87.50% / 77.78% |
| natural DEV E precision / recall | 100.00% / 66.67% | 50.00% / 66.67% |
| natural DEV 전체 정답 | 11/12 | 9/12 |
| natural FIT45 | 두 클래스 모두45/45 | 두 클래스 모두45/45 |
| strong point TRAIN6096 | 두 클래스 모두6096/6096 | 두 클래스 모두6096/6096 |

CF DEV의 동일 행 비교에서 C67이 회복한 것은2개, 손상한 것은4개로 순변화−2다. D는 회복0/손상3, E는 회복2/손상1이다. Natural DEV는 회복0/손상2이며 E 정답은 그대로다. strong/natural FIT의 완전 정답은 학습 자료 적합도이며 새로운 장면 성능의 근거가 아니다. CF DEV native1은 정답6/9→4/9이고 native4는13/18로 동일하다. native2/6/7의 이 weak 감독 범위는 여전히 없다.

[전체 수치 JSON](saved_source_comparison_v2/SOURCE_COMPARISON.json) · [DEV 9묶음 실제 crop 전부](saved_source_comparison_v2/PORTABLE_GALLERY_KO.md)

학습 전에 고정한 중심 픽셀과 부위의 차이도 별도 집계했다. 중심4 raster sample은 가리지 않지만 목표 부위를 가린 DEV COVER5개는 정답0/5→1/5였다. 목표 부위는 유지하면서 중심 sample 일부를 가린 DEV SHAM3개는 둘 다3/3이었다. 이는 물리 native점 GT가 아니라 저장 alpha에 대한 기하 진단이다. 중심 pixel만 보는 문제를 넘어 부위 관계를 안정적으로 읽었다고 할 수 없다.

## 실제 성공·실패 crop

F05 S0 N4에서는 C67만 목표 부위 COVER를 가림으로 바꾸고 BEFORE/SHAM을 보임으로 유지했다.

![목표 부위를 가릴 때만 반응한 성공](saved_source_comparison_v2/SOURCE_DEV_F05_C000000_S0_N4.png)

F05 S0 N1에서는 C67이 BEFORE와 target-disjoint SHAM까지 모두 가림으로 판단했다. 빨간 crop 중심과 청록 부위 영역도 다르다. 이 사례가 physical native 점을 가렸다는 gold는 없다.

![가리기 전부터 가림으로 판단한 실패](saved_source_comparison_v2/SOURCE_DEV_F05_C000000_S0_N1.png)

F08 S1 N4에서는 C67이 COVER를 놓치고 SHAM을 가림으로 판단했다. 단순히 가림막이 있으면 모두 E로 판정하는 것만으로 설명되지 않는 잘못된 관계 반응이다.

![COVER를 놓치고 SHAM에 반응한 실패](saved_source_comparison_v2/SOURCE_DEV_F08_C000003_S1_N4.png)

Root가 최종 V2의 9장 모두 직접 확인했다. 최초 V1 그림은 footer가 C67 표시와 겹쳐 V2에서 여백만 늘렸다. 모델·데이터·수치 재생성은 없고 숫자 파일 byte가 같다. V1 및 실행 코드는 보존했다. V1 확인 중 잘못 추정한 파일명 1개는 존재하지 않아 열지 못했으며, V2에서는 실제 갤러리 이름으로 모든9개를 확인했다.

## 검증 범위와 남은 문제

독립 QA는 생성 RGB/alpha/valid/target384PNG, q/조건/역할, 2,200개 학습 로그, 같은 초기 상태와 optimizer reset을 검사했다. 새 두 checkpoint로 CF144를 CPU 재생했고 저장 logit과 최대 차이는 C66 3.05e−5/C67 1.91e−5, 모든 판정 부호는 같았다. 이 QA PASS는 구현·저장 결과 검증이며 학습 목표 달성 판정이 아니다. old192의 클래스 조건부 RNG 전체는 old target NPZ를 다시 열지 않아 독립 재구성하지 않았고, 모든 index/mask/common equality/rolling hash를 검사했다. strong/natural 출력은 독립 집계했으나 RGB 재추론하지 않았다.

최초 trainer V1은 실행 전 검토에서 authority 문자열과 role flag의 두 schema 불일치가 발견돼 차단했다. V1을 보존하고 그 두 항목만 V2에서 수정해 실제 metadata 호환성을 확인한 후 학습했다. 실패한 모델 학습을 숨기거나 예산을 반복해 성능을 고른 것이 아니다.

이번에 해소한 것은 같은 query 조건을 유지한 부위 COVER/SHAM 데이터 제작, 실제 감독 투입, 재현 가능한 대조 비교다. **해소하지 못한 것은 새로운 SOURCE 장면·부위로의 일반화, 자연 외부 가림과의 정합성, 초기 추론 query로의 전달, 물리 native점 해석, 보류 정책**이다. SELF는 이진 E로 합치지 않았고 상류 기하 판정을 유지한다. 위치 보정·PnP는0이다. 목재/정사각형 실제 성능이 이번에 개선됐다는 새 수치는 없다.

FIT는 충분히 맞혔는데 같은 절차로 만든 SOURCE DEV에서도 실패했다. 따라서 실사 외관 전이만을 단독 원인으로 보기 어렵다. 좁고 상관된 장면/부위 수, 기존 strong point와 finite-part weak 목표 차이, native 주소 불확실성, crop 문맥 부족, shortcut/기억 중 어느 원인이 지배적인지는 아직 분리되지 않았다. 학습 update를 더 늘리는 결론도 뒷받침되지 않는다.

다음 작업은 공개 SOURCE에서 **학습 query와 실제 관찰 부위의 연결을 더 넓은 장면에서 확보하고, 동일 물체·부위가 보임/자연 외부 가림으로 바뀌는 증거와 patch 문맥을 분리 검증**하는 것이다. ROI/GT pose를 배포 입력으로 추가하거나 현재223점수로 데이터를 고르지 않는다. 원래 초기 query와 공개 GT-derived 학습 query의 차이도 별도 유지한다. 목표는 계속 진행 중이며 이번 C66/C67은 선택·채택하지 않는다.

[구현·schema repair·실제 실행·독립 QA 원문](provenance/ARCHIVE_MANIFEST.json) · [Root 결과](ROOT_RESULT.json)
'''
new(S/'PORTABLE_REPORT_KO.md',report.encode())
new(S/'ROOT_REPORT_KO.md',report.encode())
jnew(R/'V222_ROOT_PREPARATION/ROOT_ACTUAL_SOURCE_DEV9_VISUAL_REVIEW.json',{
 'completed':True,'actual_view_image':True,'version':'saved_source_comparison_v2',
 'all9_actual_paths_derived_from_gallery':True,'figures':json.loads(raw(S/'saved_source_comparison_v2/COMPLETE.json'))['SOURCE_DEV_triplet_figures'],
 'footer_separate_from_C67_label':True,'display_markers_not_training_input':True,
 'V1_preserved_layout_failure':True,'V1_wrong_guessed_filename_not_opened':'F05_C000000_S1_N1',
 'V1_actual_viewed':8,'V2_actual_viewed':9,'no_new_model_execution':True,
 'observations':['F05S0N4 correct conditionalD/E/D','F05S0N1 C67allE including BEFORE/SHAM','F05S2N1 wrong BEFORE/COVER ordering','F08S1N4 COVER miss and SHAM falseE','BlurredF08S0N4 both miss COVER'],
 'label_authority':'GT-derived parent finite-part weak, not native point gold','selected_model':None})
jnew(R/'V222_ROOT_PREPARATION/ACTUAL_RUNTIME_TERMINAL_RECEIPT.json',{
 'completed':True,'source':'root observed exec/write_stdin outputs retained in conversation; not independent process tracing',
 'producer':{'session':86815,'terminal_chunk':'f51c48','exit_code':0},
 'probe':{'session':41935,'initial_chunk':'f376c2','terminal_chunk':'65fe57','exit_code':0},
 'final_pair':{'session':98187,'initial_chunk':'56631e','terminal_chunk':'13ee9b','exit_code':0},
 'summary_V1':{'session':49610,'initial_chunk':'b7279c','terminal_chunk':'655fbc','exit_code':0},
 'summary_V2_graphic_layout_only':{'session':1199,'initial_chunk':'9f6af1','terminal_chunk':'914a3e','exit_code':0},
 'training_restarts':0,'protected_evaluation':0})
result={'completed':True,'phase':'V222','scientific_outcome':'SOURCE_GENERALIZATION_NOT_IMPROVED',
 'source_only':True,'goal_status':'active','fully_solved':False,'selected_arm':None,'adopted':False,
 'actual_final_updates':2200,'steps_each':1100,'discarded_probe_updates':64,'models':2,
 'training_complete':bind(S/'paired_source_fit_v1/COMPLETE.json'),
 'source_comparison':bind(S/'saved_source_comparison_v2/SOURCE_COMPARISON.json'),
 'independent_model_QA':bind(modelqa),'report':bind(S/'PORTABLE_REPORT_KO.md'),
 'protected_evaluation_access':0,'latest_real_benchmark_unchanged':'V220','new_PnP':0,
 'authority':'GT-derived parent query, finite-part weak; not physical native-point gold or production initializer alignment',
 'next_action':'Broaden legitimate public SOURCE query-to-observed-part/natural-occlusion evidence and distinguish context/target alignment; preserve fixed production query and evaluation-only current223. No extra epochs/seed/model selection from private scores.'}
jnew(S/'ROOT_RESULT.json',result)
print(json.dumps({'completed':True,'report':bind(S/'PORTABLE_REPORT_KO.md'),'result':bind(S/'ROOT_RESULT.json')},ensure_ascii=False))
