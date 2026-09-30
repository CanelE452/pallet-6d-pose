"""Korean decision report from frozen results, with bounded follow-up plan."""
import csv
from collections import Counter
from pathlib import Path
import torch
from . import run as C

def fmt(x):return 'NA' if x is None else f'{x:.3f}'
def tr(s):return ' / '.join(fmt(s['conditional'][k]['median']) for k in ('translation_cm','rotation_deg'))

def main():
    e1=C.read(C.DOC/'E1_SUMMARY.json');e2=C.read(C.DOC/'E2_ORACLE_SUMMARY.json');boxes=C.read(C.DOC/'E2_DETECTION_SUMMARY.json')
    e3=C.read(C.DOC/'E3_SUMMARY.json');e4=C.read(C.DOC/'E4_SUMMARY.json');e5b=C.read(C.DOC/'E5B_SUMMARY.json');e5=C.read(C.DOC/'E5_STRESS_SUMMARY.json');e6=C.read(C.DOC/'E6_SUMMARY.json')
    lineage=C.read(C.DOC/'MODEL_LINEAGE.json');lim=C.read(C.DOC/'E7_REFERENCE_LIMITS.json');cost=C.read(C.DOC/'RESOURCE_LEDGER.json');rec=C.read(C.DOC/'BY_RECORDING.json')
    follow=dict(status='PLAN_ONLY_NOT_EXECUTED',decision='다음 학습 보류; 고정12pose의 clean→자연 가림→clean 36장 짝 촬영에서 동일 가림 좌표의 clean RGB 대 가림 RGB 비교 하나',
        reason='Full125 is capable on unseen clean RGB with30px four-corner corruption (79/116 and81/116), while fixed-qO CO→OO degrades on clean29. Direction correlation is not supported as primary bottleneck. Natural99 has no paired clean views; reference and scene/crop alternatives remain.',
        sample_budget=dict(recordings=6,static_poses_per_recording=2,poses=12,images_per_pose=3,total_images=36,power_based=False),
        design='Camera/object/intrinsics fixed within triplet. New recordings selected before outputs. Same qO and clean bbox/affine/score/confidence/validity in both arms; vary only clean-before RGB vs occluded RGB. Clean-after checks static-pose assumption. Fixed PRIOR1 and FULL125; no retraining.',
        reference='Existing geometry contract using independently recorded clean-view annotations, with direct-click versus projected-point provenance recorded; no claim of independently measured physical6D. No invented occluded-corner GT. Store source images, K, dimensions, pose-hold/drift checks.',
        controls='No output-based occluder placement or resizing; one declared occluder placement per pose. Pose drift flags retained. Native OO and detection failures reported separately; no qC substitution for missing qO.',
        primary='Paired FULL125 CO vs OO T/R under same GEO and a held W/D diagnostic; report identity benefit and clean damage. Full12 denominator, failure counts, per-recording results, reference limitations.',
        decisions=dict(RGB_effect_consistent='Then prioritize a single recording/occlusion composition training comparison with fixed compute, not correlation-only corruption.',
            RGB_effect_not_consistent='Keep cause unresolved; do not launch a larger training sweep. Inspect reference/candidate/detection evidence already frozen.'),
        no_new_fits=True,no_collection_this_run=True,no_performance_guarantee=True,
        alternatives_not_selected=['direction-correlation-only retraining','new selector search','teacher-target-view3-arm sweep','source1024 inference','reference ROI inference'])
    C.save(C.DOC/'NEXT_COMPARISON.json',follow)
    questions={
      'E0':dict(status='ANSWERED',reason='HEAD/hash/actual128=29+21+78 and93∩99=92 verified; historical and current prediction coordinate parity confirmed',evidence=['RUN_MANIFEST.json','MODEL_LINEAGE.json']),
      'E1':dict(status='ANSWERED',reason='Same R0 inputs, frozen PRIOR1/FULL125 and GEO compared on99/29, including held-WD and separate OLD_REF217',evidence=['E1_SUMMARY.json','FRAME_RESULTS.csv']),
      'E1_target_accuracy':dict(status='UNRESOLVED',reason='TRAIN pseudo-target following and clean-R0 restoration measured separately. No paired clean view/actual teacher target for each natural99; physical target accuracy cannot be inferred.',evidence=['E5B_SUMMARY.json','E7_REFERENCE_LIMITS.json']),
      'E1_interpolation':dict(status='SKIPPED',reason='Endpoints already distinguish2D/pose/branch effects. Natural99 target missing; lambda sweep cannot identify target accuracy.'),
      'E1_source':dict(status='SKIPPED',reason='768 image/label/renderer bindings verified. Existing FULL125 source exports store scalar errors, not operational xy; no decision-relevant geometry gap justifies1024 new forwards.',evidence=['E7_REFERENCE_LIMITS.json']),
      'E2':dict(status='ANSWERED',reason='Same-GEO T-best/R-best whole poses and every cached detector box evaluated;8/10 T-tail frames unmatched,3 recoverable box choices and5 without matching box',evidence=['E2_ORACLE_SUMMARY.json','E2_DETECTION_SUMMARY.json']),
      'E2_wrong_instance':dict(status='UNRESOLVED',reason='Box/reference mismatch does not identify another physical instance versus an object part/background response.'),
      'E2_ROI':dict(status='SKIPPED',reason='Existing-box analysis isolates recoverable3 versus pool-missing5; changing crop/context would not resolve the selected next paired-RGB question.'),
      'E3':dict(status='ANSWERED',reason='29×2mask controlled CC/CO/OC/OO and nativeOO completed for2 refiners and identity, all58 same-object pairs; only29 images/3 recordings',evidence=['E3_SUMMARY.json','E3_PROTOCOL.json','E3_CPU_CONSISTENCY_ADDENDUM.json']),
      'E3_general_effect':dict(status='UNRESOLVED',reason='CO→OO median degradation is a within-design signal; paired recording CIs include zero,3 clusters and qC-based masks limit generalization.'),
      'E4':dict(status='ANSWERED',reason='Current99 human review174=125occluded+49visible; replaceable110+47 matched canonical corners; unknown unchanged; held/GEO separately evaluated',evidence=['E4_SUMMARY.json']),
      'E5_A':dict(status='ANSWERED',reason='Existing253 TRAIN census and fixed target following reused: hard50/2024, Full125 recovers49/50. Historical DEV93 counts not copied to99.',evidence=['../pallet_posefix_target_data_diagnosis_v1/RESULTS_KO.md']),
      'E5_B':dict(status='UNRESOLVED',reason='New fixed TRAIN29 masks yield no >20px input errors; pseudo following improves but cannot identify difficult-error scene transfer. No mask rescue.',evidence=['E5B_SUMMARY.json']),
      'E5_stress':dict(status='ANSWERED',reason='Exact marginal30px four-corner directions matched; Full125 restores79/116 independent vs81/116 correlated. No evidence here that correlation alone is the main bottleneck.',evidence=['E5_STRESS_SUMMARY.json']),
      'E6':dict(status='ANSWERED',reason='Exact REALFT_A checkpoint128 forward+sameGEO;99 medians4.937cm/2.723deg and matching98/99. Training recording overlap10/128 disclosed; improvement remains on disjoint90.',evidence=['E6_SUMMARY.json','MODEL_LINEAGE.json','E7_SUBGROUP_SUMMARY.json']),
      'E6_selection_history':dict(status='UNRESOLVED',reason='Old report calls epoch60 last; actual requested file best.pt has args epochs40, stripped epoch=-1 and train_results through16. Exact checkpoint hash known; contradictory historical selection prose not treated as fact.'),
      'E7_statistics':dict(status='ANSWERED',reason='Full denominators, failure policy, paired recording bootstrap and LORO; original metadata strata and separate natural/clean recording tables preserved',evidence=['BY_RECORDING.json','E7_SUBGROUP_SUMMARY.json']),
      'E7_noise_floor':dict(status='BLOCKED',reason='No independent repeated original clicks/annotator-repeat linkage/covariance. All1024 corner provenance fields unknown; this is not evidence all coordinates are wrong.',evidence=['E7_REFERENCE_LIMITS.json']),
      'next_intervention':dict(status='ANSWERED',reason='One36-image paired-data comparison specified; new training/collection not executed',evidence=['NEXT_COMPARISON.json'])}
    C.save(C.DOC/'QUESTION_STATUS.json',questions)
    # Serialize exact current99 error structure, rather than reusing old93 counts.
    two=C.read(C.RAW/'E1_2D_METRICS.json')['identity'];ids=C.read(C.RAW/'POPULATIONS.json')['groups']['NATURAL99'];structure=Counter();freq=Counter()
    for i in ids:
        if not two[i]['matched']:continue
        es=[v for v in two[i]['canonical_errors'] if v is not None];n=sum(v>20 for v in es)
        structure.update(corners=len(es),hard20=n,hard40=sum(v>40 for v in es),hard_frames=int(n>0),fourplus_frames=int(n>=4));freq[str(n)]+=1
    C.save(C.DOC/'CURRENT99_ERROR_STRUCTURE.json',dict(counts=dict(structure),hard_corner_count_per_frame=dict(freq),frames=99,matched=91,
        TRAIN_reference='Actual stored pseudo targets',DEV_reference='Existing evaluation annotations; errors not a common physical-GT distribution'))
    # Checkpoint internals make the E6 lineage discrepancy reviewable.
    ftck=torch.load(C.ROOT/C.read(C.RAW/'COST_E6_infer.json')['checkpoint']['path'],map_location='cpu',weights_only=False)
    C.save(C.DOC/'REALFT_A_SELECTION_HISTORY.json',dict(epoch=ftck.get('epoch'),best_fitness=ftck.get('best_fitness'),date=ftck.get('date'),
        train_args={k:ftck['train_args'].get(k) for k in ('model','data','epochs','patience','name','pretrained')},
        stored_training_epoch_count=len(ftck['train_results']['epoch']),last_stored_training_epoch=ftck['train_results']['epoch'][-1],
        checkpoint=C.read(C.RAW/'COST_E6_infer.json')['checkpoint'],conclusion='Exact checkpoint identified; old epoch60 last prose unsupported by these internals. Do not infer exact original selected epoch from stripped checkpoint.'))
    lines=['# 팔레트 6D pose: clean→가림 전이 진단 결과','',
      '**[확인] 지정 범위의 진단을 완료했다. 새 학습 0회, 고정 모델 CPU image-forward 1,027회.** 기존 FULL125·PRIOR1·R0·OLD_REF217 캐시512개를 재사용하고, 현재 full128/natural99/clean29의 같은 GEO·T/R 계약으로 재계산했다. E3, E5-B, 좌표 stress, REALFT_A 고정 추론을 수행했다. 원본 GT·체크포인트·기존 보고서·사용자 변경은 유지했다. 방법 논문 목표도 유지한다.','',
      '**[추정] 가장 강한 남은 가설은 가림 RGB에서 보정의 이득이 약해지고, 그 출력과 W/D 선택이 결합해 자연99의 작은 집계 개선을 만든다는 것이다.** 큰 T 꼬리는 별도로 검출/박스 매칭 실패에 집중한다. 방향 상관성만을 주 원인으로 삼는 학습은 이번 결과가 지지하지 않는다. 다만 RGB 비교의 recording CI는0을 포함하므로 보편적 원인으로 확정하지 않는다.','',
      '**다음 하나: 학습을 보류하고 고정12pose의 clean→가림→clean36장 짝 촬영으로, 동일 qO에서 clean RGB 대 가림 RGB 비교를 한다.** 이번 실행에서는 촬영·재클릭·새 학습을 하지 않았다. 상세는 [NEXT_COMPARISON.json](NEXT_COMPARISON.json).','',
      '## 무엇이 문제였고 어떻게 구별했나','',
      '기존 FULL125의2D 결과만으로 자연 가림6D 개선을 판단할 수 없었다. 같은 좌표를 기존 두 W/D 후보의 최종 corner8 SQPnP/LM으로 풀고, 고정된 GEO_LINEAR 선택기를 재사용했다. 보정 전 GEO가 고른 W/D를 유지하는 진단도 함께 수행했다. 동일 W/D는 동일 연속 R/t나 동일 PnP 내부 해를 뜻하지 않는다. OLD_REF217은 별도 ST 학생이므로 FULL125와의 차이를 순수 보정기 효과로 해석하지 않는다.','',
      '현재 full128=clean29+Moderate21+Severe78, natural99=Moderate21+Severe78이다. 실제 old93과 current99의 교집합은92, old-only1/current-only7이다. clean29는 ST clean78과 다르다. FULL125는 별도의 DAY253/REC_001 학습 이력이 있다. [모델 계보](MODEL_LINEAGE.json), [초기 계약](RUN_MANIFEST.json), private POPULATIONS.json에 실제 ID·SHA를 연결했다.','',
      'T는 원점 중심 cuboid의 카메라 좌표 중심 오차cm, R은 physical registry frame의C2(I,Ry180) 전체 회전 오차°다. 90°W/D 교환은 허용 대칭이 아니다. 기존 solver의 왜곡계수None, K·치수·corner 순서·중심8 보존·invalid 복원·confidence 보존을 유지했다. Pose에는 IoU gate를 추가하지 않았다. 전체/매칭 성공/공통 유효 pose 분모는 따로 보존한다.','',
      '## E1: 자연99의2D 이득과 실제T/R','',
      '| 모델+동일GEO | T 중앙값 cm | R 중앙값 ° | T P90 cm | R P90 ° | 유효/실패/미검출 | 매칭 | 매칭 코너2D 중앙값 px |','|---|---:|---:|---:|---:|---|---|---:|']
    for a in ('identity','PRIOR1','FULL125','OLD_REF217'):
        s=e1['NATURAL99']['models'][a];t=e1['NATURAL99']['twoD_matched'][a]
        lines.append(f"| {a} | {fmt(s['conditional']['translation_cm']['median'])} | {fmt(s['conditional']['rotation_deg']['median'])} | {fmt(s['conditional']['translation_cm']['P90'])} | {fmt(s['conditional']['rotation_deg']['P90'])} | {s['valid_pose']}/{s['failed_pose']}/{99-t['detected']} | {t['matched']}/99 | {fmt(t['median'])} |")
    lines += ['', 'FULL125는R0 대비2D 중앙값을10.692→8.566px로 낮췄지만, T/R 이득은 −0.417cm/−0.807°이며 P90은 줄지 않았다. PRIOR1 대비 FULL125는T −0.056cm, R +0.170°의 tradeoff다. **FULL125가 PRIOR1보다 두 pose 축 모두 우수하다는 결론은 성립하지 않는다.**', '',
      '| 경로 | identity T/R | PRIOR1 T/R | FULL125 T/R |','|---|---|---|---|',
      f"| 같은 GEO 재선택 | {tr(e1['NATURAL99']['models']['identity'])} | {tr(e1['NATURAL99']['models']['PRIOR1'])} | {tr(e1['NATURAL99']['models']['FULL125'])} |",
      f"| 보정 전 W/D 고정 | {tr(e1['NATURAL99']['models']['identity'])} | {tr(e1['NATURAL99']['models']['PRIOR1_held_identity'])} | {tr(e1['NATURAL99']['models']['FULL125_held_identity'])} |",'',
      'GEO 후보 전환은 PRIOR1 13/99, FULL125 18/99다. FULL125의 동일 프레임 동시 개선34/99, 동시 악화27/99, T만 개선23, R만 개선15다. 집계 중앙값 차이와 프레임별 차이의 중앙값은 [E1_SUMMARY.json](E1_SUMMARY.json)에 별도 저장했다.','',
      'FULL125−identity의 paired recording bootstrap95% 구간은T [−2.354,+4.942]cm, R [−12.367,+0.395]°다. 두 구간 모두0을 포함한다. 새 recording 일반화, 효과 없음, 측정 불가능 중 어느 것으로도 자동 변환하지 않는다.','',
      'clean29에서도2D 중앙값7.258→5.600px와 달리T 중앙값2.930→3.433cm로 악화한다(R 1.637→1.630°). clean 성능 저하로 clean/가림 gap이 줄어드는 것을 성공으로 세지 않았다.','',
      '## E2: 후보 선택 여지와 검출 꼬리','',
      '| 입력 | 실제GEO T/R | T-best 전체pose T/R | R-best 전체pose T/R | 한 후보가 T/R 동시 개선 가능한 프레임 |','|---|---|---|---|---:|']
    for a in ('identity','PRIOR1','FULL125','OLD_REF217'):
        o=e2['models'][a]['NATURAL99'];lines.append(f"| {a} | {tr(e1['NATURAL99']['models'][a])} | {tr(o['T_best']['summary'])} | {tr(o['R_best']['summary'])} | {e2['both_gain_counts'][a]['NATURAL99']}/99 |")
    lines += ['', '각 oracle은 후보 하나의 전체(R,t)를 선택한다. 서로 다른 후보의 최소T·최소R를 합치지 않았다. 동률은 해당 오차→hypothesis name 순이고 개선 방향 수치 동률 허용은1e−7이다. 현재 pool의 여지만 뜻하며 배포 성능이 아니다.','',
      'R0/FULL125 모두 T 상위10장 중8장이 기존IoU0.5 매칭 실패다: 맞는 박스가 저장 pool에 있으나 선택하지 않은3장, 모든 저장 박스가 참조와 미매칭인5장. 나머지2장은 매칭됐어도 코너/PnP/W-D 문제가 남는다. 미검출은0이며, 검출과 매칭은 다르다. “다른 실제 물체 검출”인지 “같은 물체 일부/배경”인지는 확정하지 않았다.','',
      'T-best로도 R0/FULL125 T P90은120.471/120.824cm로 유지된다. W/D 후보 선택만으로 큰 위치 오류가 해결되지 않는다. 기존 박스 중 참조IoU 최대 선택의 진단도 실행했으며 FULL125 T/R은11.655cm/3.626°다. 누락된 박스를 생성한 결과가 아니다. [후보 요약](E2_ORACLE_SUMMARY.json), [모든 박스 분석](E2_DETECTION_SUMMARY.json).','',
      '## E3: clean29 동일 이미지 RGB×좌표','',
      'C/O 첫 글자는RGB, 둘째는좌표다. CC=(clean,qC), CO=(clean,qO), OC=(가림,qC), OO=(가림,qO). controlled 네 조건은 clean bbox/crop/affine/score/confidence를 공유하고, nativeOO는 실제 가림 검출 경로다. qO 누락/다른 객체를 qC로 대체하지 않았고58/58쌍이 기존 박스IoU0.5 조건을 만족했다. 아래 T/R 단위는cm/°다.','',
      '| 마스크 | 모델 | CC | CO | OC | OO | nativeOO |','|---|---|---|---|---|---|---|']
    for kind in ('cover','avoid'):
        for a in ('identity','PRIOR1','FULL125'):
            s=e3[kind]['CLEAN29']['models'][a];lines.append('| '+' | '.join([kind,a]+[tr(s[c]['pose']) for c in ('CC','CO','OC','OO','nativeOO')])+' |')
    lines += ['', 'FULL125의 cover CO→OO 집계 중앙값 변화는+1.386cm/+0.790°이고, 프레임별 변화 중앙값은+0.528cm/+0.379°다. 같은 입력 좌표에서 RGB 가림이 보정을 약화시키는 관찰이다. 그러나3recording bootstrap 구간은T [−40.614,+1.396], R [−86.978,+2.082]로 매우 넓고0을 포함한다. 이 신호를 모집단 전체의 확정 원인으로 부르지 않는다. avoid에서도+0.446cm/+0.226°지만 역시 불확실성이 남는다.','',
      'FULL125 cover OO는 identity OO 대비T −0.418cm/R −0.084°지만, nativeOO와 controlledOO는T/R tradeoff다. 따라서 crop/검출 경로가 일관되게 더 나쁘다는 주장도 하지 않는다. qO−qC 평균코너오차의 프레임 중앙값은 cover5.267px/avoid4.239px로 실제 stress가 있었으나, 자연99의 큰 오류와 같은 분포는 아니다.','',
      '**마스크 해석 제한:** seed42, 기존 rectangle 크기/종횡비/8×8 noise fill 규칙을 사용하고 같은 크기·색·형태를 위치만 바꿨다. 배치는 사전 고정 qC 예측 코너 기준이다. 저장 평가 코너 위치로 점검하면 cover1장은0코너, avoid3장은1코너와 겹친다. 모든29장을 유지했고 결과를 보고 재배치하지 않았다. 따라서 완벽한 참조코너 가림/비가림 대조라고 부르지 않는다. 실제 사각형·qC/참조 중첩·RGB hash는 E3_MASK_COVERAGE.json에 있다. 가시성이 확인되지 않은 점을 실제 visible point라고 부르지 않는다.58개 조건은29개 원촬영의 반복이다.','',
      'CPU R0와 기존 GPU 캐시 사이 최대0.025px 차이가 최초 .01px 재사용 smoke 기준을 넘었다. 기존 tolerance를 완화하지 않고 clean·가림 R0와 보정기 양쪽을CPU로 재계산했다. 최초 clean smoke는 재사용했다. 따라서 E1 기존GPU 캐시와 E3 CPU clean 값은 미세하게 다르며 직접 차이를 생물학적/성능 변화로 해석하지 않는다. affine roundtrip, invalid/center 보존, 모델 state 불변을 확인했다. [E3 결과](E3_SUMMARY.json), [CPU 정합 조치](E3_CPU_CONSISTENCY_ADDENDUM.json).','',
      '## E4: 사람 가시성의 부분 참조 교체','',
      'current99의8코너792슬롯 중 사람 검토174=occluded125+visible49, unknown618이다. matched·canonical 대응·유효 좌표 조건에서 교체 가능한 것은occluded110코너/74장, visible47코너/33장이다. old93의169를 복사하지 않았다. 외부/자기 가림 subtype은 분리되지 않았다.','',
      '| 참조 교체 | GEO 재선택 T/R | baseline W/D 고정 T/R |','|---|---|---|']
    for kind in ('baseline','occluded','visible','both'):
        m=e4['groups']['NATURAL99']['models'];lines.append(f'| {kind} | {tr(m[kind+"_GEO"])} | {tr(m[kind+"_held"])} |')
    lines += ['', '가림 점의 수정과 후보 재선택이 결합하면 개선 여지가 있으나, W/D 고정R은 악화할 수 있다. unknown·중심8·미매칭 입력은 유지했다. 참조를 사용하는 진단이며 각 효과를 더해 원인 비율로 만들거나 보정기 전체의 상한이라고 해석하지 않는다. 참조점 제외PnP는 추가 의사결정을 바꾸지 않아 생략했다. [E4 결과](E4_SUMMARY.json).','',
      '## E5: 타깃 추종·전이·오류 구조','',
      '[재사용] 기존 TRAIN253의 실제 의사 타깃 기준 hard>20px는50/2024, FULL125는49/50을≤10px로 복구했다. 기존 TRAIN20/30/40px 단일점 probe를 반복하지 않았다. 이 수치는 TRAIN 타깃 추종이며 자연 참조 정확도가 아니다. [기존 보고서](../pallet_posefix_target_data_diagnosis_v1/RESULTS_KO.md).','',
      '[이번 current99] 매칭91장/702코너 중>20px203개, >40px97개, hard보유60장, 4개 이상 동시 오류26장이다. TRAIN의 타깃과DEV의 참조가 달라 두 분포를 같은 실측GT 기준이라고 해석하지 않는다. [current99 구조](CURRENT99_ERROR_STRUCTURE.json).','',
      '[E5-B] TRAIN253을ID 정렬한 등간격29장, 새 고정 마스크×2, nativeOO만174회 추론했다. 모두 같은 객체지만>20px 입력오류는0이다. stress 부족으로 어려운 오류의 전이는 미해결이다. cover의 실제 학습 의사 타깃 중앙오차는 identity1.867→FULL1250.640px로 줄지만, clean R0 복원 오차는0.638→1.829px로 늘었다. **학습 타깃 추종, clean 예측 복원, 평가 정확도는 서로 다른 기준**이다. DEV natural99의 대응 clean teacher 타깃을 새로 만들거나 가정하지 않았다. [E5-B](E5B_SUMMARY.json).','',
      '[E5 좌표 stress] clean29의 qC를 고정 복원 참조로 삼아 같은4코너·30px·RGB·crop을 유지했다. 각 코너 순번의 주변 방향 분포는 정확히 같은29개 각도를 사용하고, 독립 permutation 대 프레임 내 동일 방향으로 상관성만 바꿨다. qC는 학습 의사 타깃 또는 물리 정답이 아니다.','',
      '| 모델 | 방향 | qC 복원 중앙오차 px | 30→≤10 복구 코너 | 평가T/R |','|---|---|---:|---|---|']
    for a,x in e5['CLEAN29']['models'].items():
        for k,v in x.items():
            s=v['clean_R0_restoration'];lines.append(f"| {a} | {k} | {fmt(s['distribution']['median'])} | {s['recovered30to10']}/{s['N']} | {tr(v['pose'])} |")
    lines += ['', 'FULL125는 학습하지 않은 clean29에서도 큰4코너 오류를 상당수 복구한다. 이 결과는 용량/새 이미지 전이가 원리적으로 불가능하다는 설명에 반대 근거다. 독립79/116 대 동반81/116으로, 이 크기·코너 조합에서 방향 상관성만이 주 병목이라는 근거는 약하다. 자연가림 RGB·검출·참조 차이는 여전히 남는다. size/point-count sweep은 하지 않았다. [좌표 stress](E5_STRESS_SUMMARY.json).','',
      '## E6: 기존 실사 지도 모델 기준','',
      f"REALFT_A는 정확한 e14c59e… 체크포인트를128장CPU 추론했다. natural99 T/R={tr(e6['NATURAL99']['pose'])}, 매칭98/99이며 R0/보정기91/99보다 높다. R0 대비 bootstrap95% 구간은T [−12.974,−2.792]cm/R [−11.942,−1.372]°다. 이 고정 모델에서 개선 가능하다는 근거이며 감독 정보의 단독 인과효과는 아니다.",'',
      '학습은실사157×20, negative259×6, 합성12000; validation은합성1000이다. 초기 모델도R0 G38와 다르며 detector를 포함한 학습 구성이 다르다. 실제 이미지SHA 중복0이지만 학습recording과평가10/128(자연9/clean1)이 겹친다. 겹치지 않는natural90에서도R0 T/R11.919/5.045→REALFT_A6.141/2.822로 개선한다. 이90장도 반복 사용DEV이며 새 독립TEST가 아니다.','',
      '기존 REALFT_A report의 “epoch60 last” 설명은 요청된best.pt 내부 args epochs40, stripped epoch−1, 기록된train_results16epoch와 맞지 않는다. 이번에는 지정된 정확한hash만 사용했고 유사 모델로 대체하지 않았다. 과거 정확한 checkpoint 선택 경위는UNRESOLVED다. [계보](MODEL_LINEAGE.json), [checkpoint 내부 확인](REALFT_A_SELECTION_HISTORY.json), [중복 분리](E7_SUBGROUP_SUMMARY.json).','',
      '## E7: recording·참조 신뢰성과 반론','',
      '| 모집단 | recording | N | FULL125−identity T 중앙값차 cm | R 중앙값차 ° |','|---|---|---:|---:|---:|']
    for pop,recs in rec.items():
        for name,v in recs.items():
            d=v['paired']['FULL125']['difference_of_conditional_medians'];lines.append(f"| {pop} | {name} | {v['N']} | {fmt(d['translation_cm'])} | {fmt(d['rotation_deg'])} |")
    lines += ['', 'natural99는6recording, clean29는3recording이다. paired recording bootstrap은기존2000회/seed20260929를 재사용했고 같은draw를 모든paired모델에 적용했다. undefined draw 수를 보존하며 이번 주 비교는0회다. 모든recording의N과절대T/R·차이는[BY_RECORDING.json](BY_RECORDING.json), LORO는각결과JSON에 있다. 가림·앙각·거리·bbox크기·W/D·매칭 조건은기존metadata 범주 그대로[층별 결과](E7_SUBGROUP_SUMMARY.json)에 있다. 작은교차회귀를 추가하지 않았다.','',
      '모든128장/1024코너의source 필드가unknown이고 migration은MANUAL_REVIEW_REQUIRED다. 이것이1024좌표가틀렸다는 뜻은 아니다. manual이라는파일명/객체필드, 사람의visible/occluded판정, 직접원클릭/투영점출처, 독립물리축정답은서로다르다. geometry-resolved참조는기존2D·K·치수에서유도되며 독립장비 실측6D가아니다.','',
      '**경험적noise floor는BLOCKED:** 독립반복원클릭·annotator/repeat연결·공분산자료를 찾지못했다. 가정한2px잡음 또는CI폭으로noise floor/MDE를만들지않았다. 새재클릭/GT수정/물리앵커지표추가도하지않았다. [참조 한계](E7_REFERENCE_LIMITS.json).','',
      'Source256의image/label/renderer768개hash연결은확인했다. 기존SOURCE_BASE/FULL/SOURCE_ROWS는오차스칼라를저장하고operationalxy가없어동일출력의T/R을복원할수없다. E1/E3에서현재출력계약을직접검증했으므로1024회추가source추론은SKIPPED했다. 과거sourceR을현재C2로복사하지않았다. source좌표stress를RGB가림실험이라고부르지않는다.','',
      '## 선행연구의 원리와 이번 적용의 경계','',
      '[FixMatch](https://arxiv.org/abs/2001.07685)는약한증강의고신뢰타깃을강한증강입력과연결하고, [Noisy Student](https://arxiv.org/abs/1911.04252)는학생에noise를주면서의사라벨로학습한다. 분류결과는팔레트pose전이의보장이아니다. 이번E1/E3처럼타깃추종과실제정확도를별도로검증해야한다.','',
      '[PoseFix](https://arxiv.org/abs/1812.03595)의오류분포를반영한refinement원리를E5에연결했지만, 팔레트의W/D·검출실패를사람pose의특정오류유형과동일시하지않는다. [OpenCV PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)는대응점의pose를구하는절차다. 작은재투영잔차가올바른물체/대응점을보장하지않는것은이번박스진단에서도확인된다.','',
      '[Self6D++](https://arxiv.org/abs/2203.10339)는가림을고려한자기지도pose개선의근거다. 실제CAD/K/mask가확보되면같은후보의렌더링·마스크/RGB정합점수가참조에가까운후보를구분하는지부터검증할수있다. RGB-only경로에외부depth가필수라고가정하지않고, 렌더러를새로구축하거나논문을재학습하지않았다. 이는이번에선택한후속실험에추가하는묶음이아니다.','',
      '[SLAM-supported self-training](https://arxiv.org/abs/2203.04424)은연속관측과카메라운동정보를활용할수있다는원리다. 해당정보가있을때에만후보의다중프레임일관성과T/R관계를검증한다. 동일90°오답도시간적으로일관될수있다. SLAM구축은하지않았다.','',
      '[JCGM101](https://www.bipm.org/en/doi/10.59161/jcgm101-2008)의분포전파에는정의된입력불확실성이필요하다. 이번에는실측반복클릭분포가없어이를가정으로채우지않았다. [Cameron–Miller](https://faculty.econ.ucdavis.edu/faculty/cameron/research/Cameron_Miller_JHR_2015_February.pdf)의작은cluster수주의를따라6/3recording구간을탐색적으로표시하며특정검정의보장이나MDE로해석하지않는다.','',
      '## 다음 최소 비교: 한 가지 데이터 개입','',
      '가림RGB대cleanRGB를같은실제pose·같은qO에서비교하는짝자료를우선한다. 임시예산은새6recording×고정2pose×clean-before/occluded/clean-after=36장이다. 검정력으로정한표본수가아니다. 카메라·팔레트·K를triplet내에서고정하고occluder위치를출력확인전에한번정한다. clean-after는pose고정가정의점검용이며독립표본으로세지않는다.','',
      '고정PRIOR1/FULL125로같은qO·cleanbbox·affine·confidence를주고RGB만clean-before/occluded로바꾼다. 동일GEO와held-W/D를함께보고, nativeOO·검출실패·원래12개분모를보존한다. clean주석의직접클릭/투영출처와기하참조한계를명시한다. pose가움직인쌍은그대로표시하고성공쌍으로대체하지않는다.','',
      '이비교에서RGB가림손실이recording을넘어일관되면다음학습요인은오류방향상관성보다촬영/가림구성으로좁힌다. 일관되지않으면학습을보류하고참조·후보·검출설명을유지한다. 이번진단만으로6fit또는9fit학습을동시에제안하지않는다. ST/보정기불가능이나현재99의일반화성공을선언하지않는다.','',
      '## 질문별 완료 상태','', '| 질문 | 상태 | 근거/이유 |','|---|---|---|']
    for k,v in questions.items():lines.append(f"| {k} | {v['status']} | {v['reason']} |")
    c=cost['totals'];lines += ['', '## 실제 비용·검증·산출물','',
      f"신규 학습/optimizer update0, GPU시간0. CPU image-forward 총{c['image_forwards']:,}=E3 609(R0 87+보정기522)+E5-B174+E5stress116+REALFT_A128. E1추론0. 모델별최초smoke는최종clean출력으로재사용했다. CPU측정합{c['CPU_seconds']:.1f}초({c['CPU_seconds']/60:.1f}CPU분), stage wall합{c['summed_stage_wall_seconds']:.1f}초다. 병렬단계wall은겹치므로전체경과시간으로해석하지않는다. 단일process peak RSS {c['peak_process_RSS_KiB']/1024:.1f}MiB이며동시총메모리는미측정이다. 조사·작성·웹열람시간은stage계측에포함되지않는다.", '',
      '독립T/R 재계산2,964행, 기존R0/OLD_REF sameGEO256행parity, 원본입력/codehash, source768binding, 모델state/BN불변, 좌표roundtrip·center/invalid보존을검증했다. [VALIDATION.json](VALIDATION.json), [RESOURCE_LEDGER.json](RESOURCE_LEDGER.json), [실행 명령](EXECUTION_COMMANDS.md).','',
      '프레임·recording·모델·조건·valid/matching·T/R·2D·기준대비변화는[FRAME_RESULTS.csv](FRAME_RESULTS.csv), 후보별전체pose오차는[CANDIDATE_RESULTS.csv](CANDIDATE_RESULTS.csv)다. 전체좌표/이미지hash/변환/마스크/참조/후보상세는 `data/pallet/results/pallet_pose_diagnosis_20260930_v1/`에보존했다. 모든산출물은로컬이며commit/push/외부공유하지않았다. 표가필요한판단을담으므로장식용그림은추가하지않았다.','']
    # Keep Korean prose readable; separate frequently occurring technical tokens
    # in hand-written text without touching exact identifiers/links.
    from .report_prose import render
    C.save(C.DOC/'REPORT_KO.md',render(lines,cost['totals']))
    C.save(C.DOC/'SUMMARY.json',dict(questions=questions,next_comparison=follow,primary_natural99=e1['NATURAL99'],
        realft_natural99=e6['NATURAL99'],cost=cost['totals'],validation=C.read(C.DOC/'VALIDATION.json')))
    C.save(C.DOC/'RUN_MANIFEST_FINAL.json',dict(completed_at=C.IO.now(),initial_manifest=C.bind(C.DOC/'RUN_MANIFEST.json'),
        model_lineage=C.bind(C.DOC/'MODEL_LINEAGE.json'),actual_cost=C.bind(C.DOC/'RESOURCE_LEDGER.json'),validation=C.bind(C.DOC/'VALIDATION.json'),
        protocols=[C.bind(C.DOC/p) for p in ('E3_PROTOCOL.json','E3_CPU_CONSISTENCY_ADDENDUM.json','E5B_PROTOCOL.json','E5_STRESS_PROTOCOL.json')],
        codes=[C.bind(p) for p in sorted(C.HERE.glob('*.py'))],artifacts=[C.bind(p) for p in sorted(C.DOC.iterdir()) if p.is_file()],
        private_artifacts=[C.bind(p) for p in sorted(C.RAW.rglob('*')) if p.is_file()],
        scope_completed=True,unresolved_questions_preserved=True,new_training=0,new_collection=0,GT_edits=0,commit_or_push=False))
    print('REPORT_COMPLETE',C.DOC/'REPORT_KO.md',flush=True)

if __name__=='__main__':main()
