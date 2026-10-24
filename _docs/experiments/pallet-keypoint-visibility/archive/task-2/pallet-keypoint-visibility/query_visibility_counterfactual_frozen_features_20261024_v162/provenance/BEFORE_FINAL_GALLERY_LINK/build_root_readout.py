"""Source-only scientific readout after both fixed fits and saved-output extraction."""
from pathlib import Path
import hashlib, html, json
from datetime import datetime, timezone

V = Path(__file__).resolve().parent
ARMS = ('C43_REPLAY_C41_OWNER253_NATIVE0', 'C44_OWNER567_WITH_CF314_NATIVE0')
LABELS = ('Original owner253', 'Owner253 + controlled314')
RUN = V / 'runs/C43_C44_OWNER_CF_FIXED8000'

def read(p):
    return json.loads(p.read_text())

def bind(p):
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def pct(x):
    return '—' if x is None else f'{100*x:.1f}%'

def table(headers, rows):
    return '<table><thead><tr>' + ''.join('<th>'+html.escape(str(x))+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table>'

def main():
    trigger = read(V/'ROOT_BOTH_MODELS_COMPLETED_TRIGGER.json')
    assert trigger['completed'] and trigger['all_two_models_frozen']
    control = read(RUN/'CONTROL_COMPLETE.json')
    assert control['completed'] and control['PASS'] and set(control['arms']) == set(ARMS)
    extraction = read(V/'source_result_readout/READOUT_COMPLETE.json')
    assert extraction['completed'] and extraction['PASS']
    summaries = read(V/'source_result_readout/ARM_DATASET_SUMMARY.json')['arms']
    full = read(V/'source_result_readout/FULL_NATIVE_MATERIAL_ASPECT_TABLES.json')['arms']
    aux = read(V/'source_result_readout/ORIGINAL_GATE_AND_SEPARATE_AUXILIARY_DIAGNOSTICS.json')['arms']
    held = [summaries[a]['MR1840']['SOURCE_VALIDATION']['all_original_rows'] for a in ARMS]
    held_valid = [summaries[a]['MR1840']['SOURCE_VALIDATION']['input_valid_diagnostic'] for a in ARMS]
    train = [summaries[a]['MR1840']['SOURCE_DEVELOPMENT']['all_original_rows'] for a in ARMS]
    assert all((m['known_D'],m['known_E'],m['prediction_U_known']) == (102,52,9) for m in held)
    gates = [summaries[a]['gates'] for a in ARMS]
    e_delta = held[1]['CM_D_E_rows_D_E_U_columns'][1][1] - held[0]['CM_D_E_rows_D_E_U_columns'][1][1]
    d_delta = held[1]['CM_D_E_rows_D_E_U_columns'][0][0] - held[0]['CM_D_E_rows_D_E_U_columns'][0][0]
    results = dict(schema='V162_ROOT_SOURCE_CONTROLLED_OWNER_EXTENSION_RESULT', completed=True,
        UTC=datetime.now(timezone.utc).isoformat(), actual_feature=bind(V/'frozen_cf_features_v1/FEATURE_COMPLETE.json'),
        feature_audit=bind(V/'independent_saved_feature_review/REVIEW_COMPLETE.json'),
        actual_control=bind(RUN/'CONTROL_COMPLETE.json'), source_summary=bind(V/'source_result_readout/ARM_DATASET_SUMMARY.json'),
        source_full_tables=bind(V/'source_result_readout/FULL_NATIVE_MATERIAL_ASPECT_TABLES.json'),
        original_source_rows_each=28976, original_source_inputU_each=535, CF_rows_preserved=3816, CF_inputU=360,
        updates_each=8000, owner_rows=[253,567], additional_owner_rows=314,
        all_new_CF_native_visibility_and_quality_UNKNOWN=True, BEFORE_not_duplicated=True,
        MR_held_full=held, MR_held_input_valid_diagnostic=held_valid, MR_train_full=train, native35_passed=[g['native35_passed'] for g in gates],
        full_source_qualification=[g['original_full_prospective_source_checks_all_pass'] for g in gates],
        fixed_effect_E_true_positives_delta=e_delta, fixed_effect_D_true_positives_delta=d_delta,
        independent_entity_quality_unchanged=True, baseline_C41_exact_replay=True,
        selected_model=None, adopted=False, fully_solved=False, goal_status='active_unsolved',
        current223_reads=0, V112_reads=0, PnP_q_L_changes=0,
        limitations_KO=['공개 실사 가림은 기존 구조적 약라벨이며 공식 사람 키포인트 가시성 gold가 아니다.',
            '한 seed의 고정 대조실험이다. owner TRAIN 적합과 주 가림 일반화는 별도로 평가한다.',
            'CF는 53개 query/46개 TRAIN 프레임을 재사용했다. 477variant를 독립 실사 사진으로 세지 않는다.',
            '추가 CF anchor에는 native2/3이 없다. 원래 전체 WOOD/SQUARE/native8 범위와 평가 분모는 유지한다.',
            '별도 entity 한계와 실사 좌표품질 gold 부재가 남아 있다.'])
    out=V/'ROOT_RESULTS_KO.json'
    with out.open('x') as f:
        json.dump(results,f,ensure_ascii=False,indent=2);f.write('\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    colors=['#56718c','#178c82']
    for j,(key,title) in enumerate((('D_recall','Visible keypoint retention / full D102'),('E_recall','External occlusion recall / full E52'))):
        vals=[100*m[key] for m in held]
        ax[j].bar(range(2),vals,color=colors,width=.58)
        ax[j].set_xticks(range(2),LABELS,rotation=8);ax[j].set_ylim(0,105);ax[j].set_ylabel('%');ax[j].set_title(title)
        ax[j].grid(axis='y',alpha=.18);ax[j].set_axisbelow(True)
        for i,m in enumerate(held):
            cm=m['CM_D_E_rows_D_E_U_columns'];n=cm[j][j];d=sum(cm[j])
            ax[j].text(i,vals[i]+2,f'{n}/{d} ({vals[i]:.1f}%)',ha='center')
    fig.suptitle('Fixed source-only comparison; MR weak labels; input-U9 retained')
    fig.savefig(V/'ROOT_SOURCE_CONTROL_RESULT.png',dpi=170);plt.close(fig)
    mainrows=[]
    for a,m in zip(ARMS,held):
        mainrows.append([a.split('_')[0],m['CM_D_E_rows_D_E_U_columns'],pct(m['D_recall']),pct(m['E_precision']),pct(m['E_recall']),f"{m['prediction_U_known']}/{m['known_DE']} ({pct(m['prediction_U_rate_known'])})"])
    native_rows=[]
    validrows=[[a.split('_')[0],m['CM_D_E_rows_D_E_U_columns'],pct(m['D_recall']),pct(m['E_precision']),pct(m['E_recall']),m['prediction_U_known']] for a,m in zip(ARMS,held_valid)]
    for k in range(8):
        mm=[full[a]['MR1840'][f'SOURCE_VALIDATION::native{k}']['all_original_rows'] for a in ARMS]
        native_rows.append([k,mm[0]['known_D'],mm[0]['known_E'],*[f"{m['CM_D_E_rows_D_E_U_columns'][1][1]}/{m['known_E']}" for m in mm],mm[0]['prediction_U_known']])
    family_rows=[]
    for family,fold in [('OLD6144','VALIDATION'),('NEW20480','VALIDATION'),('PALLOX512','VAL_HALL'),('MR1840','SOURCE_VALIDATION')]:
        for a in ARMS:
            m=summaries[a][family][fold]['all_original_rows']
            family_rows.append([family,a.split('_')[0],m['known_D'],m['known_E'],pct(m['D_recall']),pct(m['E_precision']),pct(m['E_recall']),m['prediction_U_known'],m['unknown_truth']])
    owner_rows=[]
    for a in ARMS:
        x=aux[a];cm=x['owner_weak_TRAIN253_diagnostic']['CM_rows_truth_T_F_BG_columns_prediction_T_F_BG'];cf=x['CF_owner_weak_TRAIN314_diagnostic']['CM_rows_truth_T_F_BG_columns_prediction_T_F_BG']
        owner_rows.append([a.split('_')[0],x['owner_rows_used'],f'{sum(cm[i][i] for i in range(3))}/253',f'{sum(cf[i][i] for i in range(3))}/314',gates[ARMS.index(a)]['native35_passed'],gates[ARMS.index(a)]['original_full_prospective_source_checks_all_pass']])
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>고정 query 대조 데이터 추가 결과 — V162</title><style>
    body{{max-width:1180px;margin:32px auto;padding:0 22px;font:17px/1.65 system-ui;color:#183047;background:#f7f9fb}}h1{{font-size:30px}}h2{{font-size:23px;margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}th,td{{border:1px solid #d1dce4;padding:8px;vertical-align:top}}th{{background:#e8eff4}}img{{max-width:100%;height:auto}}.note{{border-left:5px solid #c28424;background:#fff5df;padding:12px 18px}}a{{color:#086a90}}code{{overflow-wrap:anywhere}}</style>
    <h1>같은 위치를 덮는 대조 데이터가 주 가림 판단을 개선했는가</h1>
    <p>기존 데이터 C43과 대조 추가 C44를 각각 seed1550·8,000 update로 학습했다. 두 모델을 모두 동결한 뒤 공개 SOURCE 성능을 비교했다. 새 모델 채택 및 전체 문제 해결은 미완료다.</p>
    <p class="note">공개 MR 검증의 외부 가림 정검출 변화: {e_delta:+d}/52. 보이는 점 정검출 변화: {d_delta:+d}/102. 입력 보류 9개를 보임 recall 분모에 포함했다. 이 수치는 공개 데이터의 기존 약라벨 기준이며 자체 평가 223장의 새 성능이 아니다.</p>
    <img src="ROOT_SOURCE_CONTROL_RESULT.png" alt="고정 공개 검증의 보임 보존과 외부 가림 검출 비교">
    <h2>주 가림 판단: 같은 원래 분모</h2>
    <p>혼동행렬은 행 D(보임)·E(외부 가림), 열 D·E·U(입력 보류) 순서다. 전체 MR 검증 360행 중 D/E 정답 154개, 그 밖의 상태 206개를 따로 보존했다.</p>
    {table(['모델','혼동행렬','보임 recall','가림 precision','가림 recall','보류율 / 알려진 D/E'],mainrows)}
    <p>입력이 유효한 145개(D93/E52)만의 조건부 진단은 아래와 같다. 위 전체 154개 결과와 분리하며, 입력 보류 9개를 지워 전체 성능으로 제시하지 않는다.</p>
    {table(['모델','입력 유효 조건부 혼동행렬','보임 recall','가림 precision','가림 recall','D/E 입력 U'],validrows)}
    <h2>무엇을 바꿨는가</h2>
    <p>원래 합성·공개 실사 28,976행과 주 손실·분할·보류·draw·optimizer·clip을 유지했다. C43은 기존 중심 표면 보조 253개, C44는 직접 덮기 157개와 옆 배치 157개를 추가한 567개를 썼다. 원본 BEFORE는 중복 학습하지 않았다. 같은 클래스별 평균 CE를 쓰므로 데이터 확장에 따라 클래스 안의 개별 행 가중치는 달라진다.</p>
    <p>새 대조 데이터는 같은 q/L에서 바뀐 fullRGB의 문맥·local96·중앙32 입력을 함께 만든다. 원본 46×8 특징 byte parity와 전체 3,816행의 전처리를 검증했다. 모든 신규 native 가림·좌표품질은 UNKNOWN으로 유지하고 별도 중심 표면 보조에만 썼다. 실패 대조 2쌍도 남겼으며 보조 손실에서는 제외했다.</p>
    <h2>보조 적합과 전체 기준을 분리</h2>
    {table(['모델','학습에 사용한 owner 수','원래 owner253 적합','CF314 분류 응답','native35 통과 수','기존 전체 SOURCE 기준 통과'],owner_rows)}
    <p>C43은 CF314를 학습에 쓰지 않았고, C44는 owner 보조 학습에 사용했다. 이 대조 입력은 모두 TRAIN 원본에서 생성됐으므로 CF 점수는 새로운 장면의 주 가림 성공을 뜻하지 않는다. 독립 entity·quality 손실은 같고 entity의 기존 한계가 남아 있다. 실제 실사 좌표품질 gold는 없다.</p>
    <h2>원래 공개·합성 검증 범위</h2>
    {table(['데이터','모델','D','E','보임 recall','가림 precision','가림 recall','D/E 입력 U','binary 밖의 정답'],family_rows)}
    <p>WOOD/SQUARE·재질·형상·native8 교차 집계는 <a href="source_result_readout/FULL_NATIVE_MATERIAL_ASPECT_TABLES.json">전체 고정 표</a>에 포함한다. 추가 실사 anchor에는 native2/3이 없고, 53query/46frame를 재사용했으므로 477variant를 독립 사진으로 세지 않는다.</p>
    <h2>MR 검증 native별 결과</h2>
    {table(['native','D','E','C43 가림 정검출/E','C44 가림 정검출/E','D/E 입력 U'],native_rows)}
    <h2>고정한 실제 crop과 저장 응답</h2>
    <p><a href="controlled_source_visual_v1/index.html">이전과 같은 공개 SOURCE12 및 대조 CF12 비교</a></p>
    <h2>해석의 범위</h2><ul>'''+''.join('<li>'+html.escape(x)+'</li>' for x in results['limitations_KO'])+'''</ul>
    <p>자체 평가223·금지 데이터·PnP·좌표보정 접근은 이번 단계에서 0회다. 원래 전체 목표는 미해결이며 모델을 선택하거나 문턱을 바꾸지 않았다.</p>
    <p><a href="ROOT_RESULTS_KO.json">요약 JSON</a> · <a href="source_result_readout/ARM_DATASET_SUMMARY.json">원래 분모·보류 포함 표</a> · <a href="source_result_readout/C44_MINUS_C43_DIAGNOSTIC.json">고정 대조 효과</a></p></html>'''
    with (V/'ROOT_READOUT_KO.html').open('x') as f:f.write(text)
    print(json.dumps({'completed':True,'results':bind(out),'report':bind(V/'ROOT_READOUT_KO.html')}))

if __name__ == '__main__':
    main()
