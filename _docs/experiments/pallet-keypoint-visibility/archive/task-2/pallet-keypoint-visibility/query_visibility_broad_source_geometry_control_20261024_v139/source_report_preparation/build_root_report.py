"""Summarize completed source control, preserving the real-evaluation boundary."""
from pathlib import Path
import hashlib,html,json,datetime
V=Path(__file__).resolve().parents[1];P=V.parent
RUN=V/'runs/C30_C31_BROAD_SOURCE_GEOMETRY_FIXED8000'
def read(p):return json.loads(Path(p).read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
r=read(V/'saved_geometry_control_diagnosis_v1/SOURCE_RESULTS.json')
c=read(RUN/'CONTROL_COMPLETE.json');a=read(V/'independent_saved_review/REVIEW.json')
visual=read(V/'saved_geometry_control_visual_report_v1/VISUAL_REPORT_COMPLETE.json')
assert all(x['completed']and x['PASS']for x in(r,c,a,visual)) and c['selected_arm_source_only']is None
corpora=('OLD_NATIVE','CONTINUOUS_NATIVE');models=('v61','C16','C29','C30','C31')
tables={co:{key:{m:r['reports'][co]['CF_PRIMARY_DECLARED'][key][m]['all_original_known_DE']for m in models}for key in('VALIDATION','VALIDATION::qerror::LE3_PX','VALIDATION::foreign_family::SAME_TARGET_MATERIAL')}for co in corpora}
summary=dict(completed=True,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_status='active',fully_solved=False,goal_turn_classification='PROGRESS',
    outcome_KO='넓힌 source에서도 G42 추가는 보임 일부만 개선하고 외부 가림은 개선하지 못했다. 동일 예산 G0 대조는 이전 C29를 완전히 재현했다. 두 모델 모두 전체 기준 미달로 미채택이다.',
    actual_execution=dict(session_id=83370,exit_code=0,arms=2,updates_each=8000,parameters_each=224513,all_jobs_terminal=True,current_forward=0,PnP=0,q_L_correction=0),
    full_and_diagnostic_tables=tables,paired_changes={co:r['joint_comparisons'][co]['VALIDATION']for co in corpora},
    eligibility={m:{k:r['gates'][m][k]for k in('passed','total','source_eligible','worst_required49_class_recall')}for m in('C30','C31')},
    selected_model=None,C30_all_outputs_training_and_final_state_exact_C29=True,
    latest_real_evaluation_unchanged=dict(model='V124_C16',D_correct=1137,D_total=1147,E_correct=35,E_total=90,development_exposed_not_blind=True),
    actual_fixed_crop_observations_KO=['두 고정4variant 묶음의 이진 판단은 C30과 C31이 같다. 원래 C16에서 이미 선택한8개 사례에서도 판단 변화는 없다.',
        'base519/native0의 빨간 외부 물체 COVER는 C29/C30/C31이 맞힌다. G42만의 새로운 성공 사례가 아니다.',
        'base590/native1의 목재 NEAR_SHAM은 GT 보임 D, 초기 오차2.540px이며 C30과 C31 모두 E로 오판한다. 보이는 간격이 있고, C31의 hidden logit은+.376→+1.009로 변했다.',
        '같은 base의 목재 COVER는 C29/C30/C31 모두 E가 맞다. C29 hidden logit은+6.0737586이다.'],
    next_action_KO='공개 팔레트 실사에서 원래 native 참조·entity·순서와 구조적 근거를 검증하는 작은 주석 가능성 검사를 먼저 수행한다. 기존 임의cue나 가상bbox8점을 native 정답으로 재명명하지 않는다.',
    limitations_KO=['이번 결과는 이 head와 학습 조건에서의 G42 비교다. 모든 구조 정보의 무용성이나 다른 모델의 불가능성을 증명하지 않는다.',
        '좌표 오차3px 이하 등 GT 구간은 진단용이다. 입력 좌표를 옮기거나 평가에서 실패를 제거하지 않았다.',
        '이 합성자료의 D/E는 등록된 코너의 보임/외부 가림이다. D 출력이 초기 query 좌표의 정확성을 보증하지 않는다.',
        '정답 자기 가림 S와 불확실 U는 외부 가림 이진 학습의 음성이 아니다. 예측 보류 U는 전체 known D/E 분모에 남겼다.',
        '같은 기본 장면과 여러 slot/variant는 상관된다. 작은 수의 개선을 독립 표본 유의성이나 실사 전이로 해석하지 않는다.',
        '현재223으로 학습·문턱·체크포인트·모델을 선택하지 않았고 새 실사 추론도 하지 않았다.'],
    bindings=dict(policy=bind(V/'ROOT_BROAD_SOURCE_GEOMETRY_POLICY_BEFORE_FIT.json'),control=bind(RUN/'CONTROL_COMPLETE.json'),
        exact_C29_replay=bind(RUN/'C30_BROAD_SOURCE_ZERO_GEOMETRY_CONTROL/C30_EXACT_C29_REPLAY.json'),
        independent_review=bind(V/'independent_saved_review/REVIEW.json'),paired_transitions=bind(V/'independent_saved_review/PAIRED_C30_C31_FULL_COHORT_TRANSITIONS.json'),
        saved_diagnosis=bind(V/'saved_geometry_control_diagnosis_v1/DIAGNOSIS_SEAL.json'),all_strata_and_cases=bind(V/'saved_geometry_control_diagnosis_v1/SOURCE_RESULTS.json'),
        source_visuals=bind(V/'saved_geometry_control_visual_report_v1/VISUAL_REPORT_COMPLETE.json'),
        public_native_readiness=bind(V/'public_native_source_readiness/PUBLIC_NATIVE_READINESS_EXISTING_METADATA_ONLY.json')))
write(V/'ROOT_RESULTS_KO.json',summary)
esc=lambda x:html.escape(str(x));pct=lambda x:'—'if x is None else f'{x*100:.2f}%'
parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V139 — 원본8점 구조 조건 대조</title>',
 '<style>body{font:17px/1.65 system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 20px;background:#f6f8fa;color:#182432}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}th,td{border:1px solid #d4dce4;padding:9px;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%;height:auto}.note{padding:18px;background:#e8f0fa;border-left:5px solid #3974af}code{font-size:13px;overflow-wrap:anywhere}a{color:#195ba0}</style>',
 '<h1>초기8점 구조 정보를 더해도 외부 가림 문제는 남았습니다</h1><p class="note">C30은 이전 C29를 학습8000회부터 최종 점수·파라미터까지 정확히 재현했습니다. C31은 같은 조건에서 native G42만 추가했습니다. 새 합성 검증에서 보임 정답은4개 늘고 외부 가림 정답은1개 줄었습니다. 두 모델 모두 미채택이며 새 실사 추론은 하지 않았습니다.</p>',
 '<p>기존 합성·공개 실사51004개, 기존 native6144개와 새 native20480개를 모두 유지했습니다. DINO 영상 encoder는 고정하고 동일한224513개 head 파라미터를 같은 초기값·표본 순서·손실·8000회·문턱0으로 학습했습니다. G42는 원본 초기8점·native번호·context/경계에서 계산한 기존42개 입력입니다. 좌표·L을 바꾸거나 정답 오차를 입력으로 사용하지 않았습니다.</p>']
def table(title,rows):
    parts.append('<h2>'+esc(title)+'</h2><p>혼동행렬: 정답 행[D,E], 예측 열[D,E,U]. 전체 known D/E 분모입니다.</p><table><tr><th>모델</th><th>혼동행렬</th><th>D recall</th><th>E recall</th><th>D precision</th><th>E precision</th><th>보류율</th></tr>')
    for m,z in rows.items():parts.append('<tr><td>'+m+'</td><td>'+esc(z['CM_D_E_rows_D_E_U_columns'])+'</td>'+''.join('<td>'+pct(z[k])+'</td>'for k in('D_recall','E_recall','D_precision','E_precision','prediction_U_rate_known'))+'</tr>')
    parts.append('</table>')
table('새 합성 검증: 보임1772 / 외부 가림130',tables['CONTINUOUS_NATIVE']['VALIDATION'])
table('기존 합성 검증: 보임828 / 외부 가림60',tables['OLD_NATIVE']['VALIDATION'])
for co,title in zip(corpora,['기존','새']):
    table(title+' 합성의 같은 재질 외부 물체', {m:tables[co]['VALIDATION::foreign_family::SAME_TARGET_MATERIAL'][m]for m in('C30','C31')})
    table(title+' 합성의 초기좌표 오차≤3px (GT 진단)', {m:tables[co]['VALIDATION::qerror::LE3_PX'][m]for m in('C30','C31')})
parts.append('<h2>원래 고정한 실제 모델 입력 crop</h2><p>예전 C16 결과로 선택한 사례를 그대로 사용했습니다. 아래96 crop은 저장한 원본 픽셀을3배 확대했으며, v61은 자체 P3 ROI의 출력을 표시합니다. 네 변형마다 실제 초기 q와 L은 다르므로 고정q 인과 실험은 아닙니다. 전체8개 기존 사례는 <a href="saved_geometry_control_diagnosis_v1/INDEX.html">상세 비교</a>에 있습니다.</p>')
for s in visual['strips']:
    rel=Path(s['strip']['path']).relative_to(V);parts.append('<img src="'+esc(rel)+'">')
parts.append('<h2>판단</h2><p>기존source에서 보임2개를 고치고1개를 잃었으며 외부 가림 변화는 없습니다. 새source에서는 보임8개를 고치고4개를 잃었고, 외부 가림1개를 잃었습니다. C31의45번째 통과 조건은 TOMIE의 보임 회복입니다. native 외부 가림이 해결된 것은 아닙니다. C30은44/64, C31은45/64로 둘 다 전체 기준 미달입니다.</p><p>실사 결과는 마지막 C16의 보임1137/1147, 외부 가림35/90으로 그대로입니다. 이 결과로 실제 평가 사진에서 새 모델이 좋아졌다고 주장하지 않습니다.</p><h2>다음 실제 작업</h2><p>'+esc(summary['next_action_KO'])+'</p><p>그 주석은 현재 평가223장과 무관한 공개 원본에서만 만듭니다. 원래 가상cuboid 참조점, 그곳의 구조적 근거와 실제 표면을 구분해야 합니다. 두 합성 형상의 참조점이 표면 꼭짓점과 일치한다는 검증은 공개 형상 전체에 적용되지 않습니다. 초기 좌표 정확성, 등록 참조의 보임, 자기 가림을 구별하고 모호한 대응을U로 남깁니다. 이번에는 공개 native 라벨을 새로 만들지 않았습니다.</p><details><summary>한계와 근거 파일</summary><ul>')
for x in summary['limitations_KO']:parts.append('<li>'+esc(x)+'</li>')
for k,b in summary['bindings'].items():parts.append('<li>'+esc(k)+': <code>'+esc(b['path'])+'</code> <code>'+esc(b['sha256'])+'</code></li>')
parts.append('</ul></details></html>')
with(V/'ROOT_READOUT_KO.html').open('x')as f:f.write('\n'.join(parts)+'\n')
print(json.dumps(dict(results=bind(V/'ROOT_RESULTS_KO.json'),report=bind(V/'ROOT_READOUT_KO.html')),ensure_ascii=False))
