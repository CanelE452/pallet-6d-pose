"""Summarize completed public-source fits only; never opens current evaluation data."""
from pathlib import Path
import hashlib
import html
import json

V = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


def bind(p):
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def pct(x):
    return '—' if x is None else f'{100*x:.1f}%'


def main():
    run = V/'runs/C2_SAME_Q_COUNTERFACTUAL_SOURCE_V1'
    baseline = V/'source_baselines/C1_FROZEN_ON_V95'
    complete = read(run/'FIT_COMPLETE.json')
    assert complete['completed'] and complete['optimizer_updates'] == 2000
    c2 = read(run/'SOURCE_RESULTS.json')
    gates = read(run/'SOURCE_GATE_RESULTS.json')
    c1 = read(baseline/'SOURCE_RESULTS.json')
    assert read(baseline/'BASELINE_COMPLETE.json')['completed']
    summary = read(V/'source_pair_intake/image_only_v2_exclusive_alpha/IMAGE_ONLY_FEASIBILITY_SUMMARY_KO.json')
    support_path=V/'source_diagnosis/QUERY_TO_AUX_SUPPORT_METADATA.json'
    support=read(support_path)
    result = dict(schema='V95_ACTUAL_PUBLIC_COUNTERFACTUAL_SOURCE_RESULTS',
                  goal_status='active', fully_solved=False,
                  C1=bind(baseline/'SOURCE_RESULTS.json'),
                  C2=bind(run/'SOURCE_RESULTS.json'),
                  C2_source_gates=bind(run/'SOURCE_GATE_RESULTS.json'),
                  construction_summary=summary,
                  current223_access=0, PnP=0, original_q_L_unchanged=True,
                  query_to_aux_support=bind(support_path),
                  source_label_limit='Conditional AI auxiliary recorded-cue-set visibility. Native expected part UNKNOWN; no human native gold.',
                  self_occlusion_learning=False,
                  selected_step=c2['selected_update'],
                  learned_checkpoint_admitted=c2['selected_update'] > 0 and gates['PASS'],
                  phase_conclusion='Source gates passed; separate frozen evaluation remains required.' if gates['PASS'] else 'Source gates failed; do not run C2 on private evaluation or select thresholds from it.')
    (V/'ROOT_RESULTS_KO.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    rows=[]
    for split in ['TRAIN','VALIDATION']:
        for model, data in [('C1 기존', c1), ('C2 선택 상태 (step '+str(c2['selected_update'])+')', c2)]:
            m=data['metrics'][split]
            for variant, label in [('BEFORE','원본'),('OPAQUE_COVER','단서 가림'),('CUE_PRESERVING_SHAM','단서 피함')]:
                x=m['by_variant'][variant]
                rows.append('<tr>'+''.join(f'<td>{html.escape(str(s))}</td>' for s in
                    [split,model,label,x['known_N'],x['matrix'],pct(x['D_recall']),pct(x['E_recall']),pct(x['known_U_rate'])])+'</tr>')
    triples=[]
    for split in ['TRAIN','VALIDATION']:
        for model, data in [('C1',c1),('C2',c2)]:
            m=data['metrics'][split]
            triples.append(f'<tr><td>{split}</td><td>{model}</td><td>{m["all3_correct"]}/{m["accepted_pairs"]}</td><td>{pct(m["all3_correct_rate"])}</td></tr>')
    grows=[]
    for name,x in gates['gates'].items():
        target='≥ '+pct(x['minimum']) if 'minimum' in x else '≤ '+pct(x['maximum'])
        grows.append(f'<tr><td>{html.escape(name)}</td><td>{pct(x["value"])}</td><td>{target}</td><td>{"PASS" if x["PASS"] else "FAIL"}</td></tr>')
    final_path=V/'source_diagnosis/FINAL2000_FIXED_NOT_SELECTOR/SOURCE_RESULTS.json'
    final_html=''
    if final_path.exists():
        final=read(final_path)
        result['fixed_final2000_diagnostic']=bind(final_path)
        fr=[]
        for split in ['TRAIN','VALIDATION']:
            x=final['metrics'][split]
            fr.append('<tr>'+''.join(f'<td>{html.escape(str(s))}</td>' for s in
                [split,x['all_known']['matrix'],pct(x['by_variant']['BEFORE']['D_recall']),pct(x['by_variant']['BEFORE']['E_recall']),pct(x['by_variant']['CUE_PRESERVING_SHAM']['D_recall']),pct(x['by_variant']['OPAQUE_COVER']['E_recall']),str(x['all3_correct'])+'/'+str(x['accepted_pairs'])])+'</tr>')
        final_html='<h2>고정한 마지막 step 2000의 실패 진단</h2><p>학습 결과를 이해하기 위해 마지막 체크포인트 하나만 별도로 공개 소스에서 읽었습니다. 모델·체크포인트·문턱을 다시 선택하거나 기존 선택 결과를 교체하지 않았습니다.</p><table><tr><th>분할</th><th>all known 혼동행렬</th><th>원본 D recall</th><th>원본 E recall</th><th>대조 D recall</th><th>가림 E recall</th><th>세 장 모두 정답</th></tr>'+''.join(fr)+'</table>'
    (V/'ROOT_RESULTS_KO.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    outcome='공개 소스 기준 통과 — 별도 고정 평가가 필요합니다.' if gates['PASS'] else '공개 소스 기준 미통과 — C2의 현재 평가 실행은 하지 않습니다.'
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V95 고정 query 가림·대조 실험</title>
<style>body{{font-family:"Nanum Gothic",sans-serif;margin:36px auto;max-width:1240px;color:#162330;line-height:1.6;padding:0 24px}}h1{{font-size:30px}}h2{{margin-top:32px}}.lead{{padding:18px;background:#edf3f8;border-left:5px solid #315a80}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border:1px solid #c9d3dc;text-align:left}}th{{background:#e7edf3}}img{{width:100%;max-width:824px;height:auto}}.small{{font-size:14px;color:#425466}}a{{color:#175ea0}}code{{word-break:break-all}}</style>
<h1>V95 · 같은 query에서 가림과 물체 존재 구분</h1>
<p class="lead"><b>{outcome}</b><br>좌표·crop 크기 고정. 현재 평가 223장 접근 0, PnP 0. 목표는 계속 진행 중입니다.</p>
<h2>실제로 수행한 실험</h2><p>공개 LOCO 160개와 Pallox 800개 원래 query를 모두 유지했습니다. 원본·가림·대조의 2,880개 항목을 보존했고, 알려진 보임 단서 143개에서 고정 YCB-V 물체 배정으로 구성을 시도했습니다. 마스크 소유가 겹친 픽셀을 제외한 새 구성에서 15쌍이 성립했습니다(Pallox 학습 9, 검증 6; LOCO 0). 첫 구성의 4쌍과 실패 기록도 보존했습니다.</p>
<p>변경한 영상 30개는 같은 좌표에서 새 P3 특징과 RGB crop을 추출했습니다. 원본과 같은 2,850개 항목은 정확히 일치하는 기존 입력을 재사용했습니다. C2는 C1과 같은 26,573개 파라미터 모델을 동일한 seed 194로 새로 초기화해 2,000회 학습했습니다. 문턱은 raw logit 0이고, 체크포인트는 공개 검증 4개 계층의 평균 BCE로만 선택했습니다(선택 step {c2['selected_update']}).</p>
<h2>같은 소스 입력에서 C1과 C2</h2><p class="small">혼동행렬: 행 [보임 D, 가림 E], 열 [보임 D, 가림 E, 보류 U]. 알려진 라벨만 포함합니다. 보류도 recall 분모에 포함하며, 원래 라벨 불명은 가림 정답으로 사용하지 않습니다.</p>
<p class="lead">C2는 step 0이 선택됐습니다. 학습 후 저장한 모든 후보의 검증 평균 BCE가 초기값 ln(2)보다 컸습니다. 아래 C2는 학습한 모델의 개선 결과가 아니라 채택 실패 기록입니다. 초기 logit 0은 고정 부호 규칙상 모두 가림으로 디코딩되므로 가림 recall 100%를 성능 개선으로 해석할 수 없습니다.</p>
<table><tr><th>분할</th><th>모델</th><th>입력</th><th>known N</th><th>혼동행렬</th><th>D recall</th><th>E recall</th><th>보류율</th></tr>{''.join(rows)}</table>
<h2>같은 쌍 세 장을 모두 맞힌 비율</h2><p>원본 D · 가림 E · 대조 D를 모두 맞혀야 성공입니다. 같은 물체가 있다는 이유만으로 양쪽을 가림으로 판단하면 실패합니다.</p><table><tr><th>분할</th><th>모델</th><th>모두 정답</th><th>비율</th></tr>{''.join(triples)}</table>
<h2>학습 전에 고정한 C2 소스 기준</h2><table><tr><th>기준</th><th>실제</th><th>요구</th><th>결과</th></tr>{''.join(grows)}</table>
{final_html}
<h2>실제 15쌍 전체 crop</h2><p>각 행은 원본 / 단서 가림 / 단서 피함 / 보호 범위와 원래 query 표시입니다. 마지막 열의 표시는 모델 입력에 들어가지 않습니다.</p>
<img src="source_pair_intake/image_only_v2_exclusive_alpha_QA/CONTACT_01_ALL15_FIXED_ORDER.png" alt="고정 순서 1~8쌍"><img src="source_pair_intake/image_only_v2_exclusive_alpha_QA/CONTACT_02_ALL15_FIXED_ORDER.png" alt="고정 순서 나머지 7쌍">
<h2>query와 보조 단서의 정렬</h2><p>전체 보임 후보 143개 중 query의 가장 가까운 native 픽셀이 보호 단서 raster 안에 있는 것은 75개였습니다. 구성에 성공한 15쌍에서는 3개뿐이며, 보호 픽셀 중심까지의 거리 중앙값은 {support['accepted_pairs']['distance_px']['median']:.2f}px였습니다. 이는 보조 단서 범위까지의 거리이며 native 정답 좌표 오차가 아닙니다.</p><p>보호 단서가 query의 RGB96 중앙 네 샘플과 겹치지 않는 것은 12/15개입니다. 실제 붙인 물체 전체는 더 넓어서, 가림 배치에서도 중앙 네 샘플이 완전히 그대로인 것은 7/15개, 단서를 피한 배치에서는 8/15개입니다. 따라서 단서를 가림/피함과 query 중심을 가림/피함은 같은 조건이 아닙니다. 이것만으로 학습 실패 원인을 단정할 수는 없지만, 다음 소스 구성은 query가 지칭하는 단서를 특정하는 라벨이 필요합니다.</p>
<h2>라벨과 해석의 한계</h2><p>이 라벨은 기록한 특정 물리 단서의 보임과 제어된 가림을 뜻하는 보조 라벨입니다. 원래 native 키포인트와의 대응은 UNKNOWN이며 사람의 native 정답 라벨이 아닙니다. 일부 단서는 query 중심에서 멀고, 가린 후에도 다른 목재가 보이거나 대조 물체가 기록하지 않은 목재를 가릴 수 있습니다. 단일 공개 bin, 검증 6쌍이므로 다양한 목재·정사각형 팔레트에 대한 해결을 입증하지 못합니다. 자기 가림 학습은 이번 단계에 포함되지 않았습니다.</p>
<p>현재 평가의 최신 고정 비교는 <a href="../query_visibility_matched_query_scalar_20261023_v94/ROOT_READOUT_KO.html">V94 C1 보고서</a>에 있습니다. 이번 단계에서는 현재 평가의 영상·라벨·점수를 읽거나 학습·문턱·체크포인트 선택에 사용하지 않았습니다.</p>
<p class="small">상세 결과: <a href="ROOT_RESULTS_KO.json">ROOT_RESULTS_KO.json</a> · 원본 좌표 유지 · PnP 없음 · 목표 진행 중</p></html>'''
    (V/'ROOT_READOUT_KO.html').write_text(page)
    print(json.dumps({'report':bind(V/'ROOT_READOUT_KO.html'),'results':bind(V/'ROOT_RESULTS_KO.json'),'C2_source_gates_PASS':gates['PASS']}))


if __name__ == '__main__':
    main()
