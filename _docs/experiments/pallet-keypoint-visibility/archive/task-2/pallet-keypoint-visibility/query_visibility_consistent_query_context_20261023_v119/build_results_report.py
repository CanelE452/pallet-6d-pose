"""Root saved-only C14 report; never executes a model or chooses a checkpoint."""
from pathlib import Path
import csv,datetime,hashlib,html,json
from collections import Counter

V=Path(__file__).resolve().parent;P=V.parent
C13=P/'query_visibility_same_material_query_head_20261023_v118'
E=V/'private_evaluation/C14_FROZEN_CURRENT223/evaluation'
R=V/'runs/C14_CONSISTENT_Q8_CONTEXT_SPATIAL_QUERY'

def bind(f):return dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
def read(f):return json.loads(f.read_text())
def csvrows(f):
    with f.open()as h:return list(csv.DictReader(h))
def key(r):return r['id'],int(r['native_corner'])
def score(f):return {key(r):r for r in csvrows(f)}

def run():
    assert read(E/'EVALUATION_SEAL.json')['PASS']
    fit=read(R/'FIT_COMPLETE.json');freeze=read(R/'MODEL_FREEZE.json');source=read(V/'public_source_comparison/RESULT.json')
    assert fit['completed']and fit['optimizer_updates']==2000 and fit['all_source_gates_pass']and freeze['selected_update']==1200
    prior=read(C13/'ROOT_RESULTS_KO.json')
    primary=list(prior['current_primary_comparison'])
    for r in csvrows(E/'D_VS_EXTERNAL_E_2X3.csv'):
        if r['cohort']!='paired_same_query_RGB' or r['model']!='V119_C14_RAW':continue
        primary.append(dict(group=r['group'],model=r['model'],display_model='C14',
            human_D_N=int(r['human_D_N']),human_E_N=int(r['human_external_E_N']),
            confusion_rows_D_E_cols_D_hidden_U=[[int(r[k])for k in ['D_pred_D','D_pred_hidden','D_pred_U']],[int(r[k])for k in ['E_pred_D','E_pred_hidden','E_pred_U']]],
            D_precision=float(r['D_precision']),D_recall=float(r['D_recall']),
            E_precision=float(r['external_E_response_precision']),E_recall=float(r['external_E_response_recall']),U_rate=float(r['unresolved_rate'])))
    order={'v61':0,'기존 실사 모델':1,'C11':2,'C12':3,'C13':4,'C14':5}
    primary.sort(key=lambda r:(r['group']!='WOOD_PRIMARY105',order[r['display_model']]))
    assert len(primary)==12
    old=score(C13/'private_evaluation/C13_FROZEN_CURRENT223/HEAD_ONLY_SCORES.csv')
    new=score(V/'private_evaluation/C14_FROZEN_CURRENT223/HEAD_ONLY_SCORES.csv')
    assert len(old)==len(new)==1784 and old.keys()==new.keys()
    for k,n in new.items():
        o=old[k]
        for field in ['query_x','query_y','roi_side']:assert float(n[field])==float(o[field]) or n[field]==o[field]
        assert n['valid']==o['valid']
    ep=read(V/'ROOT_ACTUAL_C14_SAVED_CURRENT_EVALUATION_ENTRY.json')
    truth=csvrows(Path(ep['bindings']['evaluation_manifest']['path']))
    counters={};changes=[]
    for t in truth:
        if t['paired_same_query_RGB_D_vs_E']!='True':continue
        k=key(t);label=t['human_state'];group=t['group'];o=old[k]['authoritative_decision'];n=new[k]['authoritative_decision']
        counters.setdefault(group+' / '+label,Counter())[o+'→'+n]+=1
        if o!=n:changes.append(dict(id=k[0],native_slot=k[1],group=group,human_state=label,C13=o,C14=n))
    transition={k:dict(c)for k,c in counters.items()}
    locked=read(P/'query_visibility_success_failure_visual_review_20261023_v117/SELECTION_LOCK_BEFORE_PIXELS.json')['selected']
    fixed9=[]
    for x in locked:
        k=(x['id'],x['native_slot']);a=new[k]
        fixed9.append(dict(original_native_row=x['original_native_row'],id=k[0],native_slot=k[1],human_state=x['human_state'],
            C13=old[k]['authoritative_decision'],C14=a['authoritative_decision'],C14_saved_logit=float(a['raw_query_aux_hidden_logit'])))
    result=dict(completed=True,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),version='V119_C14',goal_status='active',fully_solved=False,
        outcome_KO='문맥 입력 통일 후 외부가림 정답 목재8→9/29, 정사6/61유지. 보임599/616과531/531유지. 이 변경만으로 외부가림 문제는 해결되지 않았다.',
        actual_work=dict(source_context_views=623,source_encoder_calls=79,features=[15004,1928],occurrences=46108,
            matched_updates=2000,selected_update=1200,source_gates_pass=15,head_parameters=221825,
            unchanged_sampler_trace='ece1ac68638dc2d93a75139f209fce266b560bb707d0f12b241fe80c42e76cc9',
            current_head_calls=2,current_RGB_DINO_initializer_calls=0,current_optimizer_updates=0,PnP=0),
        current_primary_comparison=primary,current_C13_to_C14_paired_transitions=transition,current_changed_paired_rows=changes,
        fixed9_saved_C14_join=fixed9,
        current_availability=dict(total_slots=1784,valid=1739,factual_U=45,additional_U=0,paired_D_E_U_rate=0),
        manual_near_D_rows=csvrows(E/'MANUAL_NEAR_D.csv'),self_binary_response_rows=csvrows(E/'HUMAN_SELF_BINARY_RESPONSE.csv'),
        coordinate_error_rows=csvrows(E/'COORDINATE_ERROR_BINS.csv'),
        public_source_comparison=source,
        interpretation_KO=['공개 source에서 확인한 context 생성 차이만 통일한 대조다. q/L/local특징/라벨/fold/head/loss/batch/selector/문턱은 유지했다.',
            '목재 가림1점 증가는 확인됐지만 정사 가림55점·목재20점 누락이 남는다. 입력 범위 통일만으로 실사용 문제가 해결되지는 않았다.',
            'source VAL손실은C13 .032754→C14 .027326이나,119쌍 중 cover118유지/sham117→116. source손실 개선을 자연영상 가림 성능으로 대신하지 않는다.',
            'q8범위 안에 q가 들어가는 것은 구성상 성질이며, 대상·native부품·좌표 정확성의 증명이 아니다.',
            'C14는최신 source동결 비교이며 current로선택한최고모델이 아니다.'],
        current_used_for_training_normalizer_threshold_checkpoint_or_arm_choice=False,V112_used=False,
        scope_KO='자기 가림은 기존 초기 기하 경로로 별도 처리. raw hidden은 외부/자기 원인 분류가 아님. E reference좌표는추정/미검증이며 보임응답도 좌표인증이아님. 현재223과 sourceVAL은 개발노출되어새blindholdout아님.',
        completion_audit=dict(public_synthetic_small_query_head='achieved',frozen_v61_legacy_comparison='achieved',
            evaluation_only_current='achieved',external_self_coordinate_separation='achieved_with_native_coordinate_squareS_limits',
            PnP_unchanged='achieved',reliable_external_visibility='not_achieved',full_objective='incomplete'),
        evidence=dict(design=bind(V/'ROOT_C14_DESIGN_BEFORE_SOURCE_FORWARD.json'),feature_receipt=bind(V/'consistent_context_features/FEATURE_COMPLETE.json'),
            feature_integrity=bind(V/'feature_integrity_review/REVIEW.json'),fit_complete=bind(R/'FIT_COMPLETE.json'),source_freeze=bind(R/'MODEL_FREEZE.json'),
            independent_saved_source_review=bind(V/'saved_source_review/REVIEW.json'),inference=bind(V/'private_evaluation/C14_FROZEN_CURRENT223/INFERENCE_SEAL.json'),evaluation=bind(E/'EVALUATION_SEAL.json'),
            independent_saved_current_review=bind(V/'saved_current_review/REVIEW.json'),public_fixed3_visual_review=bind(V/'source_visual_followup/RESULT.json')))
    assert read(V/'saved_current_review/REVIEW.json')['PASS']
    (V/'ROOT_RESULTS_KO.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    rows=[]
    for r in primary:
        cm=r['confusion_rows_D_E_cols_D_hidden_U'];group='목재'if r['group']=='WOOD_PRIMARY105'else'정사'
        values=[group,r['display_model'],f"{cm[0][0]}/{r['human_D_N']} ({100*r['D_recall']:.1f}%)",f"{cm[1][1]}/{r['human_E_N']} ({100*r['E_recall']:.1f}%)",f"{100*r['E_precision']:.1f}%",f"{100*r['U_rate']:.1f}%",str(cm)]
        rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in values)+'</tr>')
    document=(V/'report_template.html').read_text().replace('@@PRIMARY@@',''.join(rows)).replace('@@TRANSITIONS@@',html.escape(json.dumps(transition,ensure_ascii=False,indent=2)))
    assert '@@'not in document
    (V/'ROOT_READOUT_KO.html').write_text(document)
    print(json.dumps(dict(completed=True,current_C14=[r for r in primary if r['display_model']=='C14'],transitions=transition,fixed9=fixed9),ensure_ascii=False))

if __name__=='__main__':run()
