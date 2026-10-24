"""Summarize sealed source-only results; no model, image or private-data reads."""
from pathlib import Path
import datetime, hashlib, html, json

V = Path(__file__).resolve().parents[1]
P = V.parent
S = P / 'query_visibility_continuous_native_source_20261024_v136'
RUN = V / 'runs/C28_C29_CONTINUOUS_COVERAGE_FIXED8000'
ARMS = ('C28_OLD_NATIVE_EXTRA_REPLAY_CONTROL', 'C29_CONTINUOUS_NATIVE_DATA_ADDITION')

def read(p): return json.loads(Path(p).read_text())
def bind(p): return dict(path=str(p), sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def write(p, d):
    with Path(p).open('x') as f: json.dump(d, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
def metric(cm):
    d, e = cm; ratio = lambda x,y: x/y if y else None
    return dict(CM_D_E_rows_D_E_U_columns=cm, D_recall=ratio(d[0],sum(d)), E_recall=ratio(e[1],sum(e)),
        D_precision=ratio(d[0],d[0]+e[0]), E_precision=ratio(e[1],d[1]+e[1]),
        prediction_U_rate_known=ratio(d[2]+e[2],sum(d)+sum(e)))
def percent(x): return '—' if x is None else f'{100*x:.2f}%'
def esc(x): return html.escape(str(x))

control = read(RUN/'CONTROL_COMPLETE.json')
assert control['completed'] and control['selected_arm_source_only'] is None
audit = read(V/'independent_saved_review/REVIEW.json')
assert audit['completed'] and audit['PASS']
frozen = read(S/'saved_frozen_visibility_evaluation_v1/SOURCE_RESULTS_KO.json')
baseline = read(RUN/'BASELINE_REPORT_BEFORE_FIT.json')
historic = read(P/'query_visibility_native_pair_supervision_control_20261024_v135/ROOT_RESULTS_KO.json')
arms = {a:read(RUN/a/'SOURCE_RESULTS.json') for a in ARMS}
old = {'C16':metric(baseline['native_CF']['VALIDATION']['CM_D_E_rows_D_E_U_columns']),
       'v61':metric(historic['same_actual_query_source_VAL_models']['v61']['CM_D_E_rows_D_E_U_columns'])}
new = {m:metric(frozen['summary']['CF_PRIMARY_DECLARED']['VALIDATION'][m]['all_known_DE_original_denominator']['CM_rows_D_E_cols_D_E_U']) for m in ('C16','v61')}
for name,a in zip(('C28','C29'),ARMS):
    old[name]=arms[a]['native_CF']['VALIDATION']; new[name]=arms[a]['new_native_CF']['VALIDATION']
new_near={m:metric(frozen['summary']['CF_PRIMARY_DECLARED']['VALIDATION::LE3_PX'][m]['all_known_DE_original_denominator']['CM_rows_D_E_cols_D_E_U']) for m in ('C16','v61')}
for name,a in zip(('C28','C29'),ARMS): new_near[name]=arms[a]['new_native_CF']['VALIDATION::LE3_PX']
groups={k:arms[ARMS[1]]['new_native_CF'][k] for k in arms[ARMS[1]]['new_native_CF']
        if k.startswith('VALIDATION::') and any('::'+x+'::' in k for x in ('material','aspect','foreign_family'))}
comparison=V/'saved_control_crop_comparison/COMPARISON_COMPLETE.json'
c=read(comparison); assert c['completed'] and c['PASS']
residual=V/'residual_error_diagnosis/RESULTS.json'
assert read(residual)['completed'] and read(residual)['PASS']
summary=dict(completed=True, created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    phase='V136_NATIVE_CONTINUOUS_SOURCE_AND_V137_MATCHED_DATA_COVERAGE_COMPLETE', goal_turn_classification='PROGRESS',
    goal_status='active', fully_solved=False, source_improvement_verified=True, application_performance_improved=False,
    outcome_KO='치수·시점 범위를 넓힌 자료는 같은 예산의 기존자료 반복보다 보임·가림 모두 개선했다. 전체 source 기준은 미통과이며 새 실사 평가는 하지 않았다.',
    actual_execution=dict(render_scenes=2560, native_slots=20480, initializer_frames=2560, C16_encoder_calls=2554,
        source_fit_arms=2, updates_each=8000, fit_session=41374, fit_exit_code=0, all_model_jobs_terminal=True,
        no_private_forward=True, no_PnP=True, no_query_or_L_correction=True),
    source_old_native_VAL=old, source_new_native_VAL=new, source_new_native_VAL_error_LE3=new_near,
    new_C29_subgroups=groups,
    eligibility={name:dict(PASS=sum(g['PASS'] for g in arms[a]['gates'].values()), total=64,
        source_eligible=arms[a]['source_eligible'], failed={k:g for k,g in arms[a]['gates'].items() if not g['PASS']}) for name,a in zip(('C28','C29'),ARMS)},
    selected_arm=None,
    latest_real_evaluation_unchanged=dict(model='V124_C16', D_correct=1137, D_total=1147, E_correct=35, E_total=90,
        current223_fit_threshold_checkpoint_or_arm_selection=False, development_exposed_not_blind=True),
    limitations_KO=['여기서 새 성능 향상은 합성 검증 결과다. 실사 전이는 검증되지 않았다.',
        '기존 두 형상과 절차적 재질을 확장했으며 새 토폴로지나 실사 질감의 충분성을 증명하지 않았다.',
        'D/E는 등록된 물리 코너의 보임/외부 가림이다. 초기 좌표가 실제 코너인지의 판정과는 별도다.',
        '3px 이하 구간은 GT로 나눈 사후 진단이며 배포에서 적용할 필터가 아니다. 전체 분모와 실패도 함께 보존한다.',
        'S는 자기 가림, 정답 U는 보수적인 불확실 상태다. 둘 다 외부 가림 음성으로 학습하지 않았다. 예측 보류 U와도 구별한다.',
        'v61의 학습 목표는 픽셀 존재이며 등록 코너의 외부 가림 의미와 차이가 있다.'],
    bindings=dict(control=bind(RUN/'CONTROL_COMPLETE.json'), independent_audit=bind(V/'independent_saved_review/REVIEW.json'),
        policy=bind(V/'ROOT_CONTINUOUS_COVERAGE_POLICY_BEFORE_FIT.json'),
        new_source_evaluation=bind(S/'saved_frozen_visibility_evaluation_v1/EVALUATION_SEAL.json'),
        source_data_audit=bind(S/'actual_render_saved_review/SAVED_SOURCE_AUDIT_COMPLETE.json'),
        source_RGB_alias_audit=bind(S/'actual_render_saved_review/SOURCE_NATIVE_WITNESS_RGB_ALIAS_OBSERVATIONS.json'),
        original_query_gallery=bind(S/'actual_query_visual_review/saved_gallery_v1/GALLERY_COMPLETE.json'),
        saved_control_crop_comparison=bind(comparison), residual_error_diagnosis=bind(residual),
        AST_runtime_erratum=bind(S/'ROOT_AST_RUNTIME_ERRATUM.json')))
write(V/'ROOT_RESULTS_KO.json',summary)

parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
 '<title>V137 — 자료 범위 대조 결과</title><style>body{font:17px/1.65 system-ui,sans-serif;max-width:1300px;margin:32px auto;padding:0 20px;background:#f6f8fa;color:#182432}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}th,td{border:1px solid #d4dce4;padding:10px;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%;height:auto}.note{padding:18px;background:#e8f0fa;border-left:5px solid #3974af}code{font-size:14px;overflow-wrap:anywhere}a{color:#195ba0}</style>',
 '<h1>자료 범위를 넓히면 가림 판단이 개선되는가?</h1><p class="note"><b>같은 예산의 대조군보다 보임·외부 가림 모두 개선됐습니다.</b> 전체 source 기준은 미통과입니다. 새 실사 평가는 하지 않았고 현재 실사 결과는 C16의 보임 1137/1147, 외부 가림 35/90으로 그대로입니다.</p>',
 '<p>기존 합성·공개 실사 51,004개 관측과 기존 native 합성 6,144점을 유지했습니다. 새로 640개 기본 장면 × 가림 전/실제 가림/가까운 비가림/먼 비가림 = 2,560장, 20,480점을 만들었습니다. 학습 512개·검증 128개 기본 장면은 치수·재질 씨드가 분리됐으며 두 분할 모두 높이·시점 범위를 넓혔습니다.</p>',
 '<p>C28은 기존 native 자료를 더 반복하고, C29는 같은 추가 배치 자리에 새 자료를 사용했습니다. 두 모델 모두 같은 초기 가중치·8,000회 학습·손실·문턱값 0입니다. 기존 192개와 native 64개의 매회 추출 순서는 완전히 같습니다. 새 학습 전에 세 데이터군의 C16 점수도 완전히 일치했습니다.</p>']
def table(title,models):
    parts.append('<h2>'+esc(title)+'</h2><p>행렬: 정답 행 [보임 D, 외부 가림 E], 예측 열 [D, E, 보류 U]. 자기 가림과 정답 U는 별도 집계합니다.</p><table><tr><th>모델</th><th>혼동행렬</th><th>D recall</th><th>E recall</th><th>D precision</th><th>E precision</th><th>보류율</th></tr>')
    for name in ('v61','C16','C28','C29'):
        m=models[name];parts.append('<tr><td>'+name+'</td><td>'+esc(m['CM_D_E_rows_D_E_U_columns'])+'</td>'+''.join('<td>'+percent(m[k])+'</td>' for k in ('D_recall','E_recall','D_precision','E_precision','prediction_U_rate_known'))+'</tr>')
    parts.append('</table>')
table('새 합성 검증: 보임 1,772개 / 외부 가림 130개',new)
table('기존 합성 검증도 유지: 보임 828개 / 외부 가림 60개',old)
table('새 합성 검증 중 좌표 오차 3px 이하: 보임 950개 / 외부 가림 37개',new_near)
parts.append('<p>3px 구간은 GT를 사용하는 진단입니다. 실제 입력 좌표를 보정하거나 배포에서 GT 오차로 사례를 거르지 않았습니다. 전체 검증 결과를 대체하지 않습니다.</p><h2>남은 문제</h2><p>C29는 64개 source 조건 중 44개를 통과했습니다(C28: 33개). 새 검증에서 D 오판 56개와 외부 가림 누락 18개가 남았고, 같은 재질 가림은 29/38개를 검출했습니다. 기존 검증의 같은 재질 가림은 8/16개에 머뭅니다. 보임 보존과 공개 실사 source 일부 조건도 퇴행했습니다. 두 후보 모두 미채택입니다.</p><h2>C29: 재질·모양·외부 물체별 새 검증</h2><table><tr><th>조건</th><th>혼동행렬</th><th>D recall</th><th>E recall</th><th>E precision</th><th>보류율</th></tr>')
for name,m in groups.items():
    parts.append('<tr><td>'+esc(name.removeprefix('VALIDATION::'))+'</td><td>'+esc(m['CM_D_E_rows_D_E_U_columns'])+'</td>'+''.join('<td>'+percent(m[k])+'</td>' for k in ('D_recall','E_recall','E_precision','prediction_U_rate_known'))+'</tr>')
parts.append('</table><p>여기서 재질과 종횡비는 합성 자료의 독립 요인입니다. 실사 WOOD/SQUARE 평가군의 결과는 별도입니다.</p><h2>실제 모델 입력 crop</h2><p>고정된 C16 실패·성공 선택을 그대로 사용했습니다. 원래 초기 q, L에서 저장한 RGB96이며 GT 중심으로 다시 자르지 않았습니다. 아래 비교 그림에는 마지막 C28/C29 출력도 표시합니다.</p>')
for b in c['strips']:
    path=Path(b['strip']['path']); assert path.is_file(); relative=path.relative_to(V)
    parts.append('<p><a href="'+esc(relative)+'"><img src="'+esc(relative)+'"></a></p>')
parts.append('<p>새 검증의 C29 실패 76개 중 72개가 GT 좌표 오차 3px 초과 구간에 있습니다. 이는 연관성 진단이며 좌표 오차만이 원인이라는 증거는 아닙니다. 고정 사례 중 빨간 물체에 의한 가림은 C28과 C29가 모두 고쳤으므로, 이 사례 하나를 새 자료만의 효과라고 부르지 않습니다. 목재의 가까운 비가림은 여전히 오판합니다.</p>')
parts.append('<p><a href="../'+S.name+'/actual_query_visual_review/saved_gallery_v1/INDEX.html">원본 full-image / q·GT 위치 / 96 crop / 유효 마스크 4장 보기</a></p><h2>해석과 다음 확인</h2><p>자료 범위 부족이 실제로 영향을 준다는 대조 증거를 얻었습니다. 올바른 좌표 근처의 합성 구간에서는 두 종류 판단을 함께 높일 수 있었습니다. 남은 전체 오류는 좌표 오차 구간·재질·근접 외부 물체별로 점검하며, GT 오차로 실패를 제거하지 않습니다. 다음 검증은 초기 좌표를 고치지 않고 그 좌표에서 얻는 영상·기하 정보가 남은 오류를 구분하는지 확인하는 것입니다. 현재 실사셋으로 학습·문턱·학습 시점·모델을 선택하지 않습니다.</p><details><summary>제약과 근거 파일</summary><ul>')
for s in summary['limitations_KO']:parts.append('<li>'+esc(s)+'</li>')
for k,b in summary['bindings'].items():parts.append('<li>'+esc(k)+': <code>'+esc(b['path'])+'</code> <code>'+esc(b['sha256'])+'</code></li>')
parts.append('</ul></details></html>')
out=V/'ROOT_READOUT_KO.html';out.write_text('\n'.join(parts)+'\n')
print(json.dumps(dict(results=bind(V/'ROOT_RESULTS_KO.json'), report=bind(out)),ensure_ascii=False))
