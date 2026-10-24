"""Publish the closed V184 saved-only real comparison. No model or input pixels."""
from pathlib import Path
import argparse, datetime, hashlib, html, json

V = Path(__file__).resolve().parent
MODELS = ['v61', 'legacy', 'C52', 'C53']
LABEL = {'v61':'v61', 'legacy':'기존 실사 학습 모델', 'C52':'새 RGB', 'C53':'새 RGB+SAM'}
COUNT = ['N','visible_support','occluded_support','TP','FN_decided','FP','TN',
         'U_visible','U_occluded','out_of_task_S_visible','out_of_task_S_occluded',
         'other_unresolved_visible','other_unresolved_occluded']


def bind(p):
    p=Path(p).absolute(); assert '_v112' not in str(p).lower()
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def checked(b):
    assert bind(b['path']) == b
    return Path(b['path'])


def read(p):
    return json.loads(Path(p).read_text())


def ratio(a,b):
    return None if not b else a/b


def pct(a):
    return '—' if a is None else f'{100*a:.2f}%'


def table(head,rows):
    esc=lambda x:html.escape(str(x))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def aggregate(rows,model,group='ALL'):
    rr=[r for r in rows if r['model']==model and (group=='ALL' or r['group']==group)]
    assert len(rr)==(2 if group=='ALL' else 1)
    z={k:sum(r[k] for r in rr) for k in COUNT}
    cm=[[z['TP'],z['FN_decided'],z['U_visible']+z['out_of_task_S_visible']+z['other_unresolved_visible']],
        [z['FP'],z['TN'],z['U_occluded']+z['out_of_task_S_occluded']+z['other_unresolved_occluded']]]
    assert sum(map(sum,cm))==z['N'] and sum(cm[0])==z['visible_support'] and sum(cm[1])==z['occluded_support']
    return dict(model=model,group=group,N=z['N'],D=z['visible_support'],E=z['occluded_support'],
                CM_rows_D_E_cols_D_E_U=cm,D_precision=ratio(z['TP'],z['TP']+z['FP']),
                D_recall=ratio(z['TP'],z['visible_support']),E_precision=ratio(z['TN'],z['TN']+z['FN_decided']),
                E_recall=ratio(z['TN'],z['occluded_support']),U_rate=ratio(cm[0][2]+cm[1][2],z['N']),
                E_wrongly_passed_as_visible=z['FP'],E_predicted_U=cm[1][2])


def run(entry):
    e=read(entry); assert e['approved_saved_real_root_readout'] and e['code']==bind(__file__)
    ev=read(checked(e['evaluation'])); seal=read(checked(e['prediction_seal']));vis=read(checked(e['visual']))
    assert all(x['completed'] and x['PASS'] for x in (ev,seal,vis))
    assert ev['prediction_seal']==e['prediction_seal'] and seal['model_domain_rows4288']==4288
    for b in e['reviews'].values():
        x=read(checked(b));assert x['completed'] and x['PASS']
    current_path=checked(ev['outputs']['current1784/CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json'])
    public_path=checked(ev['outputs']['publicheld360/PUBLIC_WEAK_OWNER_AND_SEPARATE_SUPPORT_ASSOCIATIONS.json'])
    c,p=read(current_path),read(public_path)
    rows=[r for r in c['tables']['metrics'] if r['cohort']=='paired_same_query_RGB' and r['task']=='D_vs_E']
    agg=[aggregate(rows,m) for m in MODELS]
    assert all((r['N'],r['D'],r['E'])==(1237,1147,90) for r in agg)
    strat=[aggregate(rows,m,g) for g in ['WOOD_PRIMARY105','SQUARE118'] for m in MODELS]
    main=[[LABEL[r['model']],pct(r['D_precision']),pct(r['D_recall']),pct(r['E_precision']),pct(r['E_recall']),pct(r['U_rate']),str(r['CM_rows_D_E_cols_D_E_U'])] for r in agg]
    groups=[[r['group'],LABEL[r['model']],f"{r['D']}/{r['E']}",pct(r['D_recall']),pct(r['E_precision']),pct(r['E_recall']),pct(r['U_rate']),r['E_wrongly_passed_as_visible']] for r in strat]
    near=[]
    for m in MODELS:
        rr=[r for r in c['tables']['manual_near_D'] if r['model']==m]
        assert len(rr)==2 and sum(r['N'] for r in rr)==376
        n=sum(r['N'] for r in rr);correct=sum(r['pred_visible'] for r in rr)
        near.append([LABEL[m],n,correct,pct(correct/n),sum(r['pred_binary_hidden'] for r in rr),sum(r['unresolved'] for r in rr)])
    public=[]
    for m in ['v61','C52','C53']:
        z=p['cohorts']['FULL360']['ALL']['models'][m]['binary_T_nonT'];t=z['per_class']['T']
        public.append([LABEL[m],str(z['known_truth_confusion']),pct(t['precision_on_known_truth']),pct(t['recall_including_prediction_U']),z['prediction_U_all'],z['known_truth'],z['unknown_truth']])
    results=dict(completed=True,PASS=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope=e['scope'],evaluation=e['evaluation'],prediction_seal=e['prediction_seal'],visual=e['visual'],reviews=e['reviews'],
        primary_all=agg,primary_by_material=strat,table_primary=main,table_material=groups,
        table_manual_near376=near,table_public_weak_center=public,
        human_state_inventory=read_json_inventory(c),original_current_slots=1784,original_current_available=1739,
        original_current_inputU=45,primary_D1147_E90_N1237=True,public_coverage=p['coverage'],
        self_table=c['tables']['human_self_binary_response'],reference_bins=c['tables']['coordinate_error_bins'],
        public_support_crop_association=p['support_crop_center_associations'],
        conclusion='Visible recall improved while true external occlusion rejection degraded sharply. Neither new model is adopted; reliable visibility/external occlusion remains unsolved.',
        both_models_evaluated_regardless_of_source_or_VAL_result=True,current_publicheld_eval_only=True,
        current_publicheld_repeated_exposure_not_blind=True,threshold_model_architecture_or_training_data_choice=False,
        PnP_qL_native_or_annotation_changes=0,selected_model=None,adopted=False,fully_solved=False,goal_status='active_unsolved')
    alignment=read(checked(e['source_target_alignment']))
    assert alignment['completed'] and alignment['PASS'] and alignment['public_CF_support_truth_all_UNKNOWN']
    results['source_target_alignment']=e['source_target_alignment']
    results['training_definition_diagnosis']={
        'source_rows_audited':alignment['rows'],
        'public_CF_support_truth_all_UNKNOWN':True,
        'definitions':alignment['definitions'],
        'causal_real_failure_explanation_established':False}
    out=V/'ROOT_RESULTS_KO.json'
    with out.open('x') as f:json.dump(results,f,ensure_ascii=False,indent=2);f.write('\n')
    css='body{font:17px/1.7 sans-serif;background:#101827;color:#e7edf8;max-width:1500px;margin:30px auto;padding:0 24px}h1,h2{line-height:1.35}a{color:#9ccaff}.note{padding:16px;background:#26364a;border-left:5px solid #ec9b61}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #42536b;padding:9px 12px;text-align:left}th{background:#22334e}.scroll{overflow:auto;margin:20px 0}img{max-width:100%;height:auto}details{padding:14px;background:#18263b;margin:20px 0}'
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>V184 동결 두 모델 실사 비교</title><style>'+css+'</style><h1>V184: 보이는 점 회수는 개선, 외부 가림 판정은 악화</h1>'
    page+='<p class="note">새 RGB 모델(C52)과 RGB+SAM 모델(C53)을 모두 고정한 뒤 같은 실사 입력에서 평가했습니다. 보이는 점 recall은 높아졌지만 실제 외부 가림을 보인다고 통과시키는 오류가 늘었습니다. 전체 성능 개선이나 목표 해결로 판단하지 않으며 어느 모델도 채택하지 않았습니다.</p>'
    page+='<h2>같은 1,237개 사람 주석에서 비교</h2><p>D=보임, E=외부 가림, U=보류입니다. 주 평가 분모는 D 1,147개·E 90개이며, 혼동행렬의 행은 정답 D/E, 열은 예측 D/E/U입니다. 원래 보류·geometry S·미해결 응답은 U로 합산하되 상세 CSV에서는 구분했습니다. 집계는 목재·정사각형의 실제 count를 합했으며 비율을 단순 평균하지 않았습니다.</p>'
    page+=table(['모델','D precision','D recall','E precision','E recall','U 비율','CM D/E → D/E/U'],main)
    page+='<p>기존 모델은 외부 가림 90개 중 83개를 E로 맞혔고 3개를 D로, 4개를 U로 냈습니다. 새 RGB는 33개를 E로 맞히고 52개를 D·5개를 U로 냈습니다. 새 RGB+SAM은 20개를 E로 맞히고 70개를 D로 냈습니다. 보임 recall 상승만으로 가림 판단이 좋아졌다고 말할 수 없습니다. v61도 보임을 대량 배제해 E recall만 높은 특성이 있어 함께 봐야 합니다.</p>'
    page+='<h2>목재와 정사각형</h2><p>전체 등록은 목재 105장·정사각형 118장입니다. 고정 주 평가에는 목재 D616/E29와 정사각형 D531/E61이 포함됩니다. 주 평가에 해당하는 정사각형 프레임은 117장이며 원 118장을 삭제한 것이 아닙니다. 전체 1,784슬롯은 별도 inventory에 유지했습니다.</p>'
    page+=table(['자료','모델','D/E 정답수','D recall','E precision','E recall','U 비율','E를 D로 오분류'],groups)
    page+='<h2>위치 오차와 자기 가림을 분리</h2><p>기존 수동 클릭 기준 오차 3px 이하의 보이는 점 376개 진단입니다. 이것은 이미 정해진 reference 연관 분석이며 새 좌표 정답이나 모델 입력이 아닙니다. 주 평가를 위치 오차·owner·품질로 걸러 유리하게 만들지 않았습니다.</p>'
    page+=table(['모델','보이는 점 수','D로 회수','D recall','E로 오분류','U'],near)
    page+='<p>사람 주석 SELF 136개는 외부 가림 주 평가에 넣지 않고 기존 self 응답 표로 보존했습니다. 두 새 모델은 SELF 분류기가 아니며 사용자의 upstream 기하 필터와 구분합니다. 키포인트를 이동하거나 PnP를 수행하지 않았습니다.</p>'
    page+='<h2>공개 실사 held 45장: 약한 중심 소속 관측</h2><p>공개 자료는 사람 native 가시성 gold가 아닌, 이전에 동결한 중심2×2의 TARGET/FOREIGN/BACKGROUND 약한 관측입니다. known 204개(T24/F96/BG84), UNKNOWN 156개를 유지했습니다. 이 표에서 C52/C53은 부가 owner argmax를 T/nonT로 비교하며, 위 사람 D/E 표의 regional support decoder와 서로 다른 출력입니다. crop 안에 팔레트가 보이는지와 중심 픽셀 소속도 구분합니다.</p>'
    page+=table(['모델','known CM T/nonT → T/nonT/U','T precision','T recall','전체 예측 U','known','정답 UNKNOWN'],public)
    page+='<p>원 입력 U는 108개, v61 자체 U는 109개입니다. 공개 held의 기존 가림 모델 저장 출력은 조사한 계보에서 찾지 못해 missing으로 표시하며 다른 자료로 대체하거나 새 추론하지 않았습니다. 공개 약한 관측의 개선을 native 외부 가림 성능으로 승격하지 않습니다.</p>'
    page+='<h2>실제 crop으로 확인</h2><p>기존에 고른 19개 사례는 원 순서를 유지했습니다. 외부 가림 그림은 고정 주 평가 E 90개 전체를 원 global index 순서로 보여줍니다. 예측이 맞거나 틀린 경우만 고르지 않았습니다. 원 RGB96 픽셀을 그대로 사용하고 숫자·주석은 바깥에 표시합니다. 전체 문맥이 부족한 crop만으로 물체 소속이나 실패의 인과 원인을 확정하지 않습니다.</p>'
    for b in e.get('main_figures',[]):
        checked(b);href=html.escape(str(Path(b['path']).relative_to(V)));page+='<a href="'+href+'"><img src="'+href+'" alt="저장 실사 수치 비교"></a>'
    page+='<details><summary>외부 가림 90개 전체 crop 보기</summary>'
    for b in e.get('external_contact_sheets',[]):
        checked(b);href=html.escape(str(Path(b['path']).relative_to(V)));page+='<a href="'+href+'"><img src="'+href+'" alt="외부 가림 원 crop 전체"></a>'
    page+='</details>'
    visualhref=html.escape(str(Path(e['visual_index']['path']).relative_to(V)));checked(e['visual_index'])
    page+='<p><a href="'+visualhref+'">고정 19개와 전체 외부 가림 gallery</a></p>'
    page+='<h2>어디까지 해결했고 무엇이 막혔나</h2><p>앞선 source TRAIN 유래 고정 401개 진단에서 작은 crop 변화 뒤 정답이 오답으로 뒤집히는 경우는 이전 각 11개에서 새 RGB 4개·RGB+SAM 3개로 줄었습니다. 이는 source 조건의 안정성 개선이며 독립 실사 성공을 뜻하지 않습니다. 이번 실사에서는 보임 회수와 외부 가림 배제를 동시에 개선하지 못했습니다.</p>'
    page+='<p>학습 자료 70,832행만 별도로 감사한 결과, 공개 실사·CF는 중심이 대상/다른 물체/배경인지에 대한 owner 학습에만 쓰였고 두 부위 support 정답은 전부 UNKNOWN이었습니다. 합성의 support는 전체 96 crop에 투영된 등록 끝부위의 보임/앞가림 존재이며 둘 다 YES일 수 있습니다. 중심 소속, crop 안 대상 부위 존재, 사람 native 가시성은 서로 다른 질문입니다. 이 정의와 직접 감독의 차이는 확인됐지만 실사 실패의 인과 원인이 확정된 것은 아닙니다.</p>'
    page+='<p>실제 그림에서도 콘과 팔레트 일부가 함께 보이고 새 모델이 D를 내는 구도가 반복됩니다. 고정 g256의 C53 응답은 부가 owner가 다른 물체(F)지만 부위 support는 보임만 YES입니다. crop 그림과 출력의 불일치는 관찰됐으나 이를 근거로 owner F를 외부 가림 정답으로 바꾸거나 decoder를 고치지 않았습니다. 다음 진단의 핵심은 학습용 공개 자료에서 사용자 목표와 같은 부위별 가림 질문을 직접 감독할 근거가 있는지 확인하는 것입니다. SAM 추가만으로 해결됐다는 증거는 없습니다.</p>'
    page+='<h2>실행과 해석의 한계</h2><p>합성·공개 TRAIN으로 학습한 최종 8,000회 가중치 두 개를 모두 사용했습니다. 이번 실사 결과 전에 모델·decoder·평가 조건을 고정했고 결과로 문턱·모델·학습 자료·구조를 선택하지 않았습니다. 원 RGB/valid 바이트, F64 q/L·native·보류, SAM 보간, 4,288개 출력 처리는 저장 감사로 확인했습니다. SAM은 초기 q8 박스만 사용한 동결 모델입니다.</p><p>현재 223장과 공개 held 45장은 이전에도 검토·평가한 자료라 fresh blind 평가라고 부르지 않습니다. 기존 가림 모델의 역사적 real-CAL 노출도 보존했습니다. 이번 C52/C53 학습·보정·문턱 선택에는 현재 자료를 쓰지 않았고 새 주석·GT 중심 crop·초기 좌표 교정·PnP는 없습니다. 자연 실사에서 보임과 외부 가림을 함께 안정적으로 판단하는 전체 목표는 아직 미해결입니다.</p>'
    chref=html.escape(str(current_path.relative_to(V)));phref=html.escape(str(public_path.relative_to(V)))
    page+='<p><a href="ROOT_RESULTS_KO.json">집계 JSON</a> · <a href="'+chref+'">사람 주석 전체 confusion·self·오차별 지표</a> · <a href="'+phref+'">공개 weak owner 전체 지표</a></p></html>'
    rp=V/'ROOT_READOUT_KO.html'
    with rp.open('x') as f:f.write(page)
    print(json.dumps(dict(completed=True,PASS=True,results=bind(out),report=bind(rp))))


def read_json_inventory(c):
    rr=[r for r in c['tables']['all1784_model_inventory'] if r['group']=='ALL']
    assert len(rr)==4 and len({r['human_state_inventory'] for r in rr})==1
    return json.loads(rr[0]['human_state_inventory'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',type=Path,required=True)
    run(p.parse_args().entry)
