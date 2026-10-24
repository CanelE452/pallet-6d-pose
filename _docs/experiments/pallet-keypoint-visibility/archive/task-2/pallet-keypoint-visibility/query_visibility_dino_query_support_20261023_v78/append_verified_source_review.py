"""Add verified saved diagnostics to the report without rerunning any model."""
from pathlib import Path
import json
R=Path(__file__).resolve().parent
def main():
 review=json.loads((R/'source_interpretation_review/INTERPRETATION_CONCLUSION_KO.json').read_text())
 qa=json.loads((R/'independent_feature_audit/INDEPENDENT_FEATURE_QA.json').read_text())
 assert review['PASS']and qa['PASS']and not review['source_gate_pass']
 fixed=review['fixed_near3_strong_center_535']
 assert fixed['V78']['V_pred_V_TP']==357 and fixed['V78']['C_pred_C_TN']==17
 assert qa['independent_fraction_cast_bit_mismatch_elements']==144
 marker='<h2>작은 좌표 오차에서도 남는 실패</h2>'
 p=R/'RESULTS_KO.html';s=p.read_text();assert marker not in s
 table='<table><tr><th>같은 공개 VAL 고정 집단</th><th>V 정답 / 459</th><th>C 정답 / 76</th></tr>'
 for model in ['V76','V77','V78']:
  row=fixed[model];table+=f'<tr><td>{model}</td><td>{row["V_pred_V_TP"]}/459 ({100*row["V_recall"]:.1f}%)</td><td>{row["C_pred_C_TN"]}/76 ({100*row["C_recall"]:.1f}%)</td></tr>'
 table+='</table>'
 text=marker+'<p>q-r≤3px·STRONG3×3 교사·참조가 중앙 셀(L∞≤1/18)에 있는 같은 공개 자연 VAL 535점을 그대로 비교했습니다. 큰 좌표 오차·crop 이탈·footprint fallback만으로 모든 잔여 오류를 설명할 수 없습니다.</p>'+table
 text+='<p>DINO 후보는 보이는 점 115개를 V77보다 더 회복했지만, 가려진 점 32개를 더 놓쳤습니다. 자연 YCB의 C recall은 TRAIN 753/1019(73.9%)→VAL 61/287(21.3%)로 크게 떨어지는 반면, 합성한 공개 가림 전후의 둘 다 정답률은 TRAIN 77.6%·VAL 78.4%로 비슷합니다. 자연 가림 분포에 집중된 일반화 실패를 보여주며, 단순한 전역 과적합이나 라벨 정의 하나가 원인이라고 확정하지 않습니다. <a href="source_interpretation_review/INTERPRETATION_CONCLUSION_KO.json">저장 출력 해석</a> / <a href="fixed_geometry_source_diagnosis/PUBLIC_NATURAL_FIXED_GEOMETRY_SAVED_COMPARISON.json">같은 좌표 집단 검산</a></p>'
 text+='<p>미리 고른 실제 crop 4쌍은 이번 DINO 후보가 모두 맞혔습니다. 그럼에도 자연 가림 전체 recall은 21.3%이므로, 이 네 그림을 전체 전이 성공의 증거로 해석할 수 없습니다. 조건부 RGB 변화 반응과 자연 가림 일반화를 분리해야 합니다.</p>'
 text+='<h2>입력 수치 검산</h2><p>전 10,002행의 q/L/context와 17개 입력 속성은 원본과 정확히 일치했습니다. 동결 특징의 반복 추출은 일치하며, 저장 특징은 유한한 FP16입니다. 독립 FP64 canvas 비율과 GPU FP32에서 저장한 FP16 비율은 240,048개 중 144개에서 한 단계 차이가 났고 최대 절대 차이 0.00024414로 사전 전체 비율 bound 0.00048928 안에 들었습니다. 0 유효 영역 판정 변화는 없습니다.</p>'
 text+='<p>별도 독립 검산의 최초 1e-6 token-weight 허용값은 실패 기록으로 보존했습니다. 이후 FP32 보간·평균 연산의 오차 상한을 도출하고 CPU/GPU 계산을 대조했습니다. 입력·라벨·모델·문턱·source 성능 기준은 수정하지 않았습니다. 이 수치 검산을 분류 성능 성공으로 해석하지 않습니다. <a href="independent_feature_audit/INDEPENDENT_FEATURE_QA.json">독립 특징 검산</a> / <a href="independent_preparation/VALIDITY_NUMERIC_BOUND_DERIVATION.json">연산 오차 상한</a></p>'
 target='<h2>검산과 저장 파일</h2>';assert s.count(target)==1;s=s.replace(target,text+target)
 p.write_text(s)
 print(p)
if __name__=='__main__':main()
