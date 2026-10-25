"""Format already-saved scalar summary; preserve all original actual join bytes."""
from pathlib import Path
import json,hashlib,stat
H=Path(__file__).absolute().parent;O=H.parent/'ycb_pair_join_v1'
def read(p):
 if any(s in str(p).casefold() for s in ('v112','__pycache__','.pyc')):raise ValueError('Forbidden before FS')
 for a in reversed((p,*p.parents)):
  if stat.S_ISLNK(a.lstat().st_mode):raise ValueError('Symlink')
 return p.read_bytes()
def bind(p):
 b=read(p);return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}
s=json.loads(read(O/'SUMMARY.json'));a=s['groups']['ALL_YCB'];g=s['groups']['FIXED_REVIEW_FLAGGED'];other=s['groups']['OTHER_YCB'];ex=s['family_CF_exposure']
text=f'''# 기존 YCB SOURCE FIT CF 행동·학습 draw 진단\n\n기존 저장 scalar 자료를 row_id와 원 sample_index로 결합했습니다. 새 이미지·배열·모델·GT 읽기, forward, 학습은 없습니다.\n\n| 그룹 | 성공쌍 | COVER E | SHAM D | 두 역할 모두 정답 | COVER logit > SHAM |\n|---|---:|---:|---:|---:|---:|\n| 전체 YCB | {a['successful_pairs']} | {a['COVER_E']} | {a['SHAM_D']} | {a['COVER_E_and_SHAM_D']} | {a['COVER_gt_SHAM']} |\n| 고정 주목 조합: obj7 aux3/7 또는 obj2 aux2 | {g['successful_pairs']} | {g['COVER_E']} | {g['SHAM_D']} | {g['COVER_E_and_SHAM_D']} | {g['COVER_gt_SHAM']} |\n| 나머지 YCB | {other['successful_pairs']} | {other['COVER_E']} | {other['SHAM_D']} | {other['COVER_E_and_SHAM_D']} | {other['COVER_gt_SHAM']} |\n\n주목 조합은 전체{g['attempts']}시도 중 성공{g['successful_pairs']}, 실패 U {g['failed_U']}입니다. 성공143쌍은 모두 두 역할에서 실제 학습 draw가 있었습니다. COVER {g['COVER_actual_gradient_draws']}회 중 정답 {g['COVER_draw_weighted_correct']}회, SHAM {g['SHAM_actual_gradient_draws']}회 중 정답 {g['SHAM_draw_weighted_correct']}회입니다. 이는 반복 draw 분모이며 고유 pair 수와 다릅니다.\n\n전체 YCB 성공 pool830쌍 중 COVER는{a['COVER_unique_pairs_drawn']}개 고유쌍, SHAM은{a['SHAM_unique_pairs_drawn']}개를 실제 사용했습니다. 둘 중 적어도 한 역할 사용은{a['unique_pairs_ever_drawn_either_role']}쌍, 양 역할 사용은{a['unique_pairs_both_roles_drawn']}쌍입니다. SYN 성공 pool{ex['0']['COVER_pool']}쌍에서는 COVER {ex['0']['COVER_unique_drawn']}, SHAM {ex['0']['SHAM_unique_drawn']}개만 실제 draw됐습니다. 따라서 pool의 모든 역할 이미지가 실제 gradient로 사용됐다고 해석하지 않습니다.\n\n이 결과는 기존의 합성 SOURCE FIT pair에 대한 고정 head 기능과 draw 노출 진단입니다. 자연 SOURCE DEV나 current 성능, 실패 원인·인과관계의 증명이 아닙니다. 기존 metadata의 과거 모델 점수·GT reference·owner 값은 이번 join에 사용하지 않았습니다. YCB native는 UNKNOWN이며 associated-r 약한 교사와 literal q 중심 상태를 구분합니다.\n\n[저장 SUMMARY](../ycb_pair_join_v1/SUMMARY.json) · [group metrics](../ycb_pair_join_v1/GROUP_METRICS.csv) · [YCB 전체841행](../ycb_pair_join_v1/ALL841_YCB_PAIR_ROWS.csv) · [5406 실제 draw 노출](../ycb_pair_join_v1/ALL5406_ACTUAL_DRAW_EXPOSURE.csv)\n\n원 실행의 혼합 언어 보고서는 보존했습니다. 이 문서는 같은 저장 summary를 한국어로 표시한 교정본이며 숫자·원 CSV·원 COMPLETE·원 join code를 변경하지 않았습니다.\n'''
p=H/'KO.md'
with p.open('x') as f:f.write(text)
receipt={'completed':True,'PASS':True,'wording_only':True,'original_complete':bind(O/'COMPLETE.json'),'original_report_preserved':bind(O/'KO.md'),'original_summary':bind(O/'SUMMARY.json'),'current_KO_report':bind(p),'repair_code':bind(Path(__file__).absolute()),'original_artifacts_modified':False,'new_metric_reduction_or_model_forward':False,'new_SOURCE_payload_reads':0}
with (H/'COMPLETE.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'COMPLETE':bind(H/'COMPLETE.json'),'report':bind(p)},ensure_ascii=False))
