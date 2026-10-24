"""Additive Korean interpretation of already sealed source saved metrics only."""
from pathlib import Path
import hashlib,json

HERE=Path(__file__).resolve().parent
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def load(p):return json.loads(p.read_text())
def write(p,d):
 with p.open('x')as f:f.write(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
r=load(HERE/'SOURCE_INTERPRETATION_REVIEW.json')
assert r['PASS']and r['best_update']==100 and not r['source_candidate_gate_pass']
cohorts={x['cohort']:x['metrics']for x in r['natural_TRAIN_VAL_and_after']if x['model']=='V78'}
tr=cohorts['NATURAL_ORIGINAL|YCB_PUBLIC_REAL|train'];va=cohorts['NATURAL_ORIGINAL|YCB_PUBLIC_REAL|val']
ctr=cohorts['CONTROLLED_AFTER|YCB_PUBLIC_REAL|train'];cva=cohorts['CONTROLLED_AFTER|YCB_PUBLIC_REAL|val']
pair={x['cohort']:x for x in r['paired_TRAIN_VAL']if x['model']=='V78'}
strict=r['fixed_strict_cohort']
conclusion=dict(
 completed=True,PASS=True,review_scope='분석 재현/봉인 PASS이며 후보 모델의 source gate PASS가 아님',
 best_update=100,threshold=.5,source_gate_pass=False,gate_pass_count=sum(v['PASS']for v in r['source_gate_checks'].values()),gate_fail_count=sum(not v['PASS']for v in r['source_gate_checks'].values()),
 summary_KO='고정 DINO 후보는 공개 source의 보이는 support 보존과 인공 가림 전후 구분을 개선했지만, 자연 held-out YCB의 COVERED 구분에 실패했다. 분포에 따라 실패 양상이 달라지며 전체 문제 해결이나 실제 human semantic 전이를 입증하지 못했다.',
 natural_public_TRAIN_VAL=dict(TRAIN=tr,VAL=va,C_recall_VAL_minus_TRAIN=va['covered_recall']-tr['covered_recall'],V_recall_VAL_minus_TRAIN=va['visible_recall']-tr['visible_recall']),
 controlled_public_TRAIN_VAL=dict(TRAIN=ctr,VAL=cva,pair_TRAIN=pair['YCB_PUBLIC_REAL|train'],pair_VAL=pair['YCB_PUBLIC_REAL|val'],pair_both_VAL_minus_TRAIN=pair['YCB_PUBLIC_REAL|val']['both_correct_rate']-pair['YCB_PUBLIC_REAL|train']['both_correct_rate']),
 fixed_near3_strong_center_535=strict,
 findings_KO=[
  '자연 공개 YCB의 C recall은 TRAIN753/1019=73.90%에서 VAL61/287=21.25%로52.64%p 감소한다. 반면 V recall은84.50%에서82.90%로1.60%p 감소한다. 자연 검증의 특정 covered 집단에 일반화 실패가 집중된다는 관측이다.',
  '가림 개입 pair 양쪽 정답은 TRAIN2402/3095=77.61%, VAL573/731=78.39%로 약한 TRAIN–VAL 격차를 보이지 않는다. 같은 after 이미지의 C anchor와 V 비anchor를 동시에 맞힌 query 쌍도 VAL726/948=76.58%다. 따라서 모든 공개 실사 관측을 처리하지 못한다거나 모든 개입 검증이 과적합으로 붕괴했다는 설명은 자료와 맞지 않는다.',
  '같은 input q/L/box에서 자연 C와 새로 덮은 C가 다른 결과를 보이므로, 현재 fixed candidate의 실패는 자연 C와 인공 C의 분포 차이에 특이적이다. 다만 이 차이가 texture/경계/occluder 모양/target 정체성/관측가능성/teacher proxy 상관 중 무엇 때문인지는 분리하지 않았다.',
  '고정된 near≤3px·STRONG_3X3·native9 중심cell 교집합에서 V는357/459=77.78%(FN102), C는17/76=22.37%(false-visible59)다. V77의 V242/459보다115개 회복했지만 C49/76보다32개 놓쳤다. 개선은 가림과 보임을 함께 정확히 구분한 완전한 성공이 아니라 명확한 tradeoff다.',
  '이 strict cohort의 남은 오류 전부를 큰 초기 좌표 오차, ref가 crop 바깥인 MISS, footprint fallback 라벨 탓으로 돌릴 수 없다. 반대로 조건부 expected-support teacher가 물리적 human semantic corner visibility와 같다는 주장도 성립하지 않는다. 두 한계를 동시에 유지한다.',
  'TRAIN–VAL 자연 C 격차는 일반화 실패와 일치하지만 그 자체가 순수한 memorization/overfitting의 인과 증거는 아니다. 장면·물체구성·자연 가림분포·proxy label 특성이 split과 함께 달라진다. 인공 pair 결과가 좋다는 사실도 seam shortcut 부재나 자연 외부가림 인과 이해를 증명하지 않는다.',
  'V76/V77/V78은 동일 source 교사와 동일 ref/query cohort를 평가하므로 정의 차이만으로 후보 간 성능 변화를 설명할 수 없다. 그러나 DINO·pooling·head·초기화가 함께 달라졌으므로 변화의 원인을 DINO 표현 하나로 귀속할 수도 없다.',
  '사전 4조건 중 자연 V recall과 인공 pair 양쪽 정답 2개만 통과했다. 자연 C precision17.09%와 C recall21.25%가 기준에 미달하므로 source gate는 HOLD다. 다른 checkpoint/threshold를 재선택하지 않으며 현재223 비교도 수행하지 않는다.',
 ],
 not_concluded=['과적합 하나가 모든 실패의 인과 원인','teacher 정의만이 모든 실패의 원인','DINO 하나가 변화의 인과 원인','자연 가림의 target ownership/semantic E/S를 학습','초기 좌표 jitter 내성 검증','현재 평가 223에 대한 전이 성공'],
 bindings=[bind(HERE/'SOURCE_INTERPRETATION_REVIEW.json'),bind(HERE/'REVIEW_SEAL.json'),bind(Path(__file__))],
 RGB_mask_feature_weights_current_reads=0,new_forward_fit_download_PnP=0,model_threshold_checkpoint_selection_changes=False,fully_solved=False,
)
write(HERE/'INTERPRETATION_CONCLUSION_KO.json',conclusion)
write(HERE/'FINAL_INTERPRETATION_SEAL.json',dict(completed=True,PASS=True,source_gate_pass=False,artifacts=[bind(HERE/'SOURCE_INTERPRETATION_REVIEW.json'),bind(HERE/'REVIEW_SEAL.json'),bind(HERE/'INTERPRETATION_CONCLUSION_KO.json'),bind(Path(__file__))],current_forward_fit_PnP=0))
print(json.dumps(bind(HERE/'FINAL_INTERPRETATION_SEAL.json')))
