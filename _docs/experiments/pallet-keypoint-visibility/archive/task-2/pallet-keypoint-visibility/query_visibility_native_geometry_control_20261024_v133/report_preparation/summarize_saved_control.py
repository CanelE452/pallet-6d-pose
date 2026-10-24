"""Saved C24/C25 source results only; no model/checkpoint/RGB/private I/O."""
from pathlib import Path
import argparse, hashlib, html, json, traceback
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent
POLICY=V/'ROOT_MATCHED_GEOMETRY_POLICY_BEFORE_INPUTS.json'
RUN=V/'runs/C24_C25_MATCHED_G42_FIXED8000'
PROTOCOL=H/'SAVED_REPORT_PROTOCOL.json';OUT=V/'saved_control_report_v1'
ARMS=('C24_ZERO_GEOMETRY_CONTROL','C25_NATIVE_GEOMETRY42')
SHORT={ARMS[0]:'C24',ARMS[1]:'C25'}
FAMILIES=('SAME_TARGET_MATERIAL','BACKGROUND_LIKE','CARDBOARD_TAN','HIGH_CONTRAST')
FAMILY_KO={'SAME_TARGET_MATERIAL':'같은 재질','BACKGROUND_LIKE':'배경과 비슷한 가림','CARDBOARD_TAN':'갈색 종이 상자','HIGH_CONTRAST':'강한 색 대비'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def same(a,b):return a['path']==b['path']and a['sha256']==b['sha256']
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def pct(v):return '—'if v is None else '%.1f%%'%(100*float(v))
def clean_float(v):return float(v)if np.isfinite(v)else None
def pred_name(v):return {-1:'보류',0:'보임',1:'외부 가림'}[int(v)]
def decode(raw,available):return np.where(available,(raw>=0).astype(np.int8),-1).astype(np.int8)

def cm_metric(pred,rows,mask):
    y=np.array([{'D':0,'E':1}.get(r['raw_state'],-1)for r in rows],np.int8)
    cm=np.array([[int((mask&(y==t)&(pred==p)).sum())for p in(0,1,-1)]for t in(0,1)],np.int64)
    ratio=lambda a,b:float(a/b)if b else None
    return dict(CM_D_E_rows_D_E_U_columns=cm.tolist(),D_recall=ratio(cm[0,0],cm[0].sum()),E_recall=ratio(cm[1,1],cm[1].sum()),
                D_precision=ratio(cm[0,0],cm[:,0].sum()),E_precision=ratio(cm[1,1],cm[:,1].sum()),
                prediction_U_known=int(cm[:,2].sum()),prediction_U_rate_known=ratio(cm[:,2].sum(),cm.sum()),
                prediction_U_all=int((mask&(pred==-1)).sum()),prediction_U_rate_all=ratio((mask&(pred==-1)).sum(),mask.sum()),
                all_original_rows=int(mask.sum()),known_DE=int(cm.sum()),
                teacher_S_outside_external=int(sum(take and r['raw_state']=='S'for take,r in zip(mask,rows))),
                teacher_U_outside_external=int(sum(take and r['raw_state']=='U'for take,r in zip(mask,rows))))

def scopes(rows):
    val=np.array([r['split']=='VALIDATION'for r in rows]);nb=~np.array([r['convention_boundary']for r in rows],bool)
    family=np.array([r['factors']['foreign_family']for r in rows])
    masks={'VALIDATION':val,'VALIDATION::NONBOUNDARY':val&nb,
           'VALIDATION::NONBOUNDARY::SAME_TARGET_MATERIAL':val&nb&(family=='SAME_TARGET_MATERIAL')}
    for name in FAMILIES:masks['VALIDATION::foreign_family::'+name]=val&(family==name)
    return masks

def authorize(entrypath):
    e=read(entrypath)
    assert e['approved_saved_report']is True
    assert all(e[k]is False for k in('model_forward','fit','checkpoint_body_read','RGB_read','current223','V112','PnP','label_edit','automatic_retry'))
    p=read(PROTOCOL);assert same(e['code'],bind(__file__))and same(e['protocol'],bind(PROTOCOL))and same(p['code'],e['code'])
    check(p['policy']);check(p['control_complete'])
    control=read(p['control_complete']['path'])
    assert control['completed']and control['PASS']and control['final_update8000_both']and control['same_all8000_old_native_draws']
    assert control['application_solved']is False and control['new_private_current_V112_PnP_q_correction']==0
    assert same(control['policy'],p['policy'])and same(control['code'],p['trainer_code'])and same(control['protocol'],p['trainer_protocol'])
    assert set(control['arms'])==set(ARMS)and not OUT.exists()
    result={}
    for arm in ARMS:
        fit=read(check(control['arms'][arm]['fit']));freeze=read(check(control['arms'][arm]['freeze']))
        assert fit['completed']and fit['optimizer_updates']==8000 and fit['arm']==arm and fit['draw_sha256']==control['draw_sha256']
        assert freeze['completed']and freeze['selected_update']==8000 and not freeze['current_allowed']and not freeze['V112_allowed']
        assert same(fit['results'],p['arms'][arm]['results'])and same(fit['freeze'],control['arms'][arm]['freeze'])
        parity=read(check(fit['initialization_parity']));assert parity['completed']and parity['PASS']
        r=read(check(fit['results']));assert r['completed']and r['arm']==arm and r['selected_update']==8000 and not r['VAL_checkpoint_choice']
        assert len(r['gates'])==42 and r['source_eligible']==all(g['PASS']for g in r['gates'].values())
        assert r['source_eligible']==control['arms'][arm]['source_eligible']and r['current_V112_PnP']==0
        assert r['same_draws_both_arms']and r['no_native_paired_hinge']and r['old_native_all8000_draw_sha256']==control['draw_sha256']
        result[arm]=r
    assert set(result[ARMS[0]]['gates'])==set(result[ARMS[1]]['gates'])
    # No checkpoint binding in freeze is opened or hashed.
    return e,p,control,result

def saved_rows(p,results):
    es=read(check(p['baseline_evaluation_seal']));assert es['completed']and es['PASS']and same(es['scores'],p['baseline_rows'])
    rows=[json.loads(x)for x in check(p['baseline_rows']).read_text().splitlines()if x]
    assert len(rows)==6144 and all(r['global_index']==i and r['scene_index']==i//8 and r['native_slot']==i%8 for i,r in enumerate(rows))
    ids=np.array([r['query_id']for r in rows]);q=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in rows],np.float64)
    L=np.array([r['roi_side']if r['roi_side']is not None else np.nan for r in rows],np.float64)
    states=np.array([r['raw_state']for r in rows]);splits=np.array([r['split']for r in rows])
    basepred=np.array([{'D':0,'E':1,'U':-1}[r['C16']['prediction']]for r in rows],np.int8)
    raw={'C16':np.array([r['C16']['raw_logit']if r['C16']['raw_logit']is not None else np.nan for r in rows],np.float32)}
    pred={'C16':basepred};available={'C16':np.array([r['C16']['available']for r in rows],bool)}
    for arm in ARMS:
        name=SHORT[arm]
        with np.load(check(p['arms'][arm]['native_scores']),allow_pickle=False)as f:
            for key,expected in(('global_index',np.arange(6144)),('occurrence_id',ids),('split',splits),('raw_teacher_state_CF',states)):
                assert np.array_equal(f[key],expected),(arm,key)
            assert np.array_equal(f['query_xy'],q,equal_nan=True)and np.array_equal(f['L'],L,equal_nan=True)
            raw[name]=np.array(f['raw_head_logit_before_U'],copy=True);pred[name]=np.array(f['prediction'],copy=True);available[name]=np.array(f['query_available'],copy=True)
            assert np.array_equal(f['query_logit'],np.where(available[name],raw[name],0).astype(np.float32))
        assert raw[name].shape==pred[name].shape==available[name].shape==(6144,)
        assert np.array_equal(pred[name],decode(raw[name],available[name]))and np.array_equal(available[name],available['C16'])
    masks=scopes(rows);metrics={}
    baseline=read(check(p['baseline_report']))
    for group,mask in masks.items():
        metrics[group]={m:cm_metric(pred[m],rows,mask)for m in('C16','C24','C25')}
        for name,reported in [('C16',baseline['native_CF'][group])]+[(SHORT[a],results[a]['native_CF'][group])for a in ARMS]:
            actual=metrics[group][name]
            for key in('CM_D_E_rows_D_E_U_columns','prediction_U_known','all_original_rows','known_DE','teacher_S_outside_external','teacher_U_outside_external'):
                assert actual[key]==reported[key],(group,name,key)
            for key in('D_precision','D_recall','E_precision','E_recall','prediction_U_rate_known'):
                assert actual[key]==reported[key],(group,name,key)
    return rows,pred,raw,metrics

def select_cases(rows,pred):
    valid=lambda r:r['split']=='VALIDATION'and not r['convention_boundary']and r['raw_state']in('D','E')
    samewood=[r for r in rows if valid(r)and r['raw_state']=='E'and r['factors']['foreign_family']=='SAME_TARGET_MATERIAL']
    differences=[r for r in rows if valid(r)and r['error_px']is not None and r['error_px']<=3 and pred['C24'][r['global_index']]!=pred['C25'][r['global_index']]]
    cases=[];seen=set()
    for category,pool in(('같은 재질 · 대응 경계 제외 · 외부 가림',samewood),('3px 이하 · 두 모델의 판단 차이',differences)):
        for r in pool:
            key=(r['base_index'],r['native_slot'])
            if key in seen or len(cases)>=8:continue
            seen.add(key);cases.append(dict(category=category,selected_index=r['global_index'],base_index=key[0],native_slot=key[1],
                companion_indices=[(key[0]*4+i)*8+key[1]for i in range(4)]))
    return dict(cases=cases,same_material_nonboundary_E_rows=len(samewood),maximum_cases=8,RGB_read=False,
                actual_q_can_differ_across_variants=True,selection_changes_no_labels_or_training=True)

def html_table(headers,body):
    esc=lambda x:html.escape(str(x))
    return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>'for x in row)+'</tr>'for row in body)+'</tbody></table>'

def run(entrypath):
    e,p,control,results=authorize(entrypath)
    rows,pred,raw,metrics=saved_rows(p,results)
    selection=select_cases(rows,pred)
    OUT.mkdir()
    try:
        gates={SHORT[a]:dict(passed=sum(bool(v['PASS'])for v in results[a]['gates'].values()),total=42,
                             failed=[k for k,v in results[a]['gates'].items()if not v['PASS']],eligible=results[a]['source_eligible'])for a in ARMS}
        allval=metrics['VALIDATION']
        baseline=allval['C16'];c24=allval['C24'];c25=allval['C25']
        outcome='두 모델 모두 사전 기준을 통과하지 못해 채택하지 않았습니다.'if control['selected_arm_source_only']is None else SHORT[control['selected_arm_source_only']]+'가 공개·합성 자료의 사전 기준을 통과했습니다. 현재 실사 평가의 성공 확인은 아직 남아 있습니다.'
        delta_E=c25['E_recall']-c24['E_recall'];delta_D=c25['D_recall']-c24['D_recall']
        if delta_E==0 and delta_D<0:
            finding='구조 정보를 추가한 C25는 외부 가림 재현율이 C24와 같고, 보임 재현율은 더 낮았습니다.'
        else:
            finding='구조 정보의 추가 효과는 아래 같은 입력·학습 횟수의 C24/C25 결과로 비교했습니다.'
        status=(finding+' C24는 보임 '+pct(c24['D_recall'])+'·외부 가림 '+pct(c24['E_recall'])+', C25는 보임 '+pct(c25['D_recall'])+'·외부 가림 '+pct(c25['E_recall'])+'입니다. '+
                '사전 기준은 C24 '+str(gates['C24']['passed'])+'/42, C25 '+str(gates['C25']['passed'])+'/42를 통과했습니다. '+outcome+' 현재 평가셋은 이번 학습·선택에 사용하지 않았습니다.')
        facts=[]
        for case in selection['cases']:
            rr=[]
            for i in case['companion_indices']:
                r=rows[i]
                rr.append(dict(global_index=i,variant=r['variant'],raw_state=r['raw_state'],query_xy=r['query_xy'],reference_xy=r['reference_xy'],
                               L=r['roi_side'],error_px=r['error_px'],convention_boundary=r['convention_boundary'],foreign_family=r['factors']['foreign_family'],
                               models={m:dict(prediction=int(pred[m][i]),raw_hidden_logit=clean_float(raw[m][i]),available=bool(pred[m][i]!=-1))for m in('C16','C24','C25')}))
            facts.append(dict(case,rows=rr))
        report=dict(completed=True,PASS=True,source='PUBLIC_REAL_REPLAY_AND_SYNTHETIC_NATIVE_EVALUATION',application_solved=False,
                    status_KO=status,control_complete=p['control_complete'],models=results,metrics=metrics,gate_counts=gates,
                    selected_arm_source_only=control['selected_arm_source_only'],case_selection=selection,cases=facts,
                    source_native_metrics_independently_recomputed=True,actual_model_calls=0,checkpoint_body_reads=0,RGB_reads=0,
                    new_private_access=0,labels_edited=False,stress_rerun=False,geometry_impossibility_claim=False)
        write(OUT/'REPORT_RESULTS_KO.json',report)
        with(OUT/'STATUS_KO.txt').open('x')as f:f.write(status+'\n')
        main=[]
        group_names={'VALIDATION':'合성 검증 전체','VALIDATION::NONBOUNDARY':'합성 검증 · 대응 경계 제외',
                     'VALIDATION::NONBOUNDARY::SAME_TARGET_MATERIAL':'같은 재질 · 대응 경계 제외'}
        group_names['VALIDATION']='합성 검증 전체'
        for group,label in group_names.items():
            for model in('C16','C24','C25'):
                m=metrics[group][model];cm=m['CM_D_E_rows_D_E_U_columns']
                main.append([label,model,html.escape(str(cm)),pct(m['D_precision'])+' / '+pct(m['D_recall']),pct(m['E_precision'])+' / '+pct(m['E_recall']),
                             pct(m['prediction_U_rate_known'])+' / '+pct(m['prediction_U_rate_all'])])
        family=[]
        for name in FAMILIES:
            group='VALIDATION::foreign_family::'+name
            for model in('C16','C24','C25'):
                m=metrics[group][model];cm=m['CM_D_E_rows_D_E_U_columns']
                family.append([FAMILY_KO[name],model,str(cm[0][0])+'/'+str(sum(cm[0]))+' ('+pct(m['D_recall'])+')',str(cm[1][1])+'/'+str(sum(cm[1]))+' ('+pct(m['E_recall'])+')'])
        gate_rows=[]
        for key in results[ARMS[0]]['gates']:
            label=key.replace('OLD14::','기존 자료 유지 · ').replace('HISTORICAL15::','기존 회귀 검사 · ').replace('NATIVE::','초기 좌표 합성 평가 · ')
            cells=[]
            for arm in ARMS:
                g=results[arm]['gates'][key]
                value='통과'if g['PASS']else'실패'
                if 'value'in g and isinstance(g['value'],(int,float)):value+=' ('+pct(g['value'])+' / 기준 '+pct(g.get('limit'))+')'
                cells.append(value)
            gate_rows.append([html.escape(label)]+cells)
        cases_html=''
        for n,c in enumerate(facts,1):
            body=[]
            for r in c['rows']:
                error='—'if r['error_px']is None else'%.2f'%r['error_px']
                state={'D':'보임','E':'외부 가림','S':'자기 가림','U':'불확실'}[r['raw_state']]
                model_cells=[]
                for model in('C16','C24','C25'):
                    z=r['models'][model];score='미추론'if z['raw_hidden_logit']is None else'%.3f'%z['raw_hidden_logit']
                    model_cells.append(pred_name(z['prediction'])+' ('+score+')')
                body.append([html.escape(r['variant']),state,error,html.escape(str(r['query_xy'])),('—'if r['L']is None else'%.2f'%r['L'])]+model_cells)
            cases_html+='<details><summary>사례 '+str(n)+' · '+html.escape(c['category'])+' · 기본 장면 '+str(c['base_index'])+' / 채널 '+str(c['native_slot'])+'</summary>'+html_table(['장면','정답 정의','좌표 오차 px','초기 q','crop L','C16','C24','C25'],body)+'</details>'
        table=html_table(['범위','모델','혼동행렬 D/E × 예측 D/E/보류','보임 정밀도 / 재현율','외부 가림 정밀도 / 재현율','보류율 D/E / 전체'],main)
        page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>초기 키포인트 구조 정보 대조 결과</title><style>body{font-family:system-ui,sans-serif;background:#f7f9fc;color:#172235;margin:28px}main{max-width:1400px;margin:auto}p{line-height:1.75}table{border-collapse:collapse;width:100%;background:#fff;margin:16px 0;font-size:14px}th,td{border:1px solid #d9dfeb;padding:9px;text-align:left}th{background:#eaf0f8}details{margin:16px 0}summary{cursor:pointer;line-height:1.7}.note{background:#fff1d8;padding:16px;border-radius:8px}a{color:#2359a6}</style><main><h1>초기 키포인트 구조 정보 대조 결과</h1><p class="note">'''+html.escape(status)+'''</p><p>C24는 구조 입력을 0으로 고정했고, C25는 초기 키포인트의 채널과 상대 배치 42개 값을 사용했습니다. 나머지 입력, 모델 크기, 학습 순서와 8,000회 학습 횟수는 같습니다. 공개 실사와 합성 자료로 학습했으며, 아래 초기 좌표 평가는 합성 장면에서 얻은 결과입니다.</p><h2>초기 추론 좌표에서의 판단</h2><p>행은 보임·외부 가림, 열은 보임·외부 가림·입력 보류입니다. 보류도 재현율 분모에 포함했습니다. 자기 가림과 정답 불확실은 외부 가림 정답과 분리했습니다. 대응 경계 제외 결과도 native 의미 대응의 완전한 인증을 뜻하지 않습니다.</p>'''+table+'<h2>외부 가림 재질별 결과</h2>'+html_table(['가림 종류','모델','보임 맞힘 / 전체','외부 가림 맞힘 / 전체'],family)+'<h2>사전에 정한 42개 기준</h2><p>C24 '+str(gates['C24']['passed'])+'/42, C25 '+str(gates['C25']['passed'])+'/42. 모든 기준을 통과해야 다음 평가 대상이 됩니다.</p><details><summary>42개 기준 펼치기</summary>'+html_table(['기준','C24','C25'],gate_rows)+'</details><h2>같은 기본 장면의 가림 전후</h2><p>같은 재질의 외부 가림 사례를 우선 포함했습니다. 네 장면에서 초기 q를 각각 다시 추론했으므로 q와 L이 달라질 수 있습니다. 같은 좌표를 고정한 대조가 아니며, 표의 점수는 외부 가림 logit입니다. 0 이상이면 외부 가림으로 판단합니다.</p>'+cases_html+'<p><a href="'+html.escape(str(OUT/'REPORT_RESULTS_KO.json'))+'">전체 수치와 사례 근거</a> · <a href="'+html.escape(p['control_complete']['path'])+'">완료 기록</a></p></main></html>'
        hp=OUT/'READOUT_KO.html'
        with hp.open('x')as f:f.write(page)
        write(OUT/'REPORT_SEAL.json',dict(completed=True,PASS=True,entry=bind(entrypath),code=e['code'],protocol=e['protocol'],
                                        results=bind(OUT/'REPORT_RESULTS_KO.json'),status=bind(OUT/'STATUS_KO.txt'),HTML=bind(hp),
                                        actual_model_RGB_checkpoint_or_private_reads=0,browser_rendered=False))
        print(json.dumps(dict(completed=True,seal=bind(OUT/'REPORT_SEAL.json'),status_KO=status)),flush=True)
    except BaseException:
        write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

def prepare():
    policy=read(POLICY);control_path=RUN/'CONTROL_COMPLETE.json';control=read(control_path)
    assert control['completed']and control['PASS']and same(control['policy'],bind(POLICY))
    ev=read(check(policy['native_evaluation']))
    p=dict(completed=True,code=bind(__file__),policy=bind(POLICY),control_complete=bind(control_path),
           trainer_code=control['code'],trainer_protocol=control['protocol'],baseline_evaluation_seal=policy['native_evaluation'],
           baseline_rows=ev['scores'],baseline_report=bind(RUN/'BASELINE_REPORT_BEFORE_FIT.json'),arms={},output=str(OUT),
           maximum_case_base_slots=8,source_scope='Public real replay + synthetic native source evaluation; no current performance claim',
           no_model_checkpoint_RGB_private_or_stress_forward=True)
    for arm in ARMS:
        fit=read(check(control['arms'][arm]['fit']))
        p['arms'][arm]=dict(results=fit['results'],native_scores=bind(RUN/arm/'FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz'))
    # Metadata-only preparation hashes saved score files but does not load them.
    fixture=[dict(raw_state=s)for s in('D','E','D','S')]
    m=cm_metric(np.array([0,0,-1,1],np.int8),fixture,np.ones(4,bool))
    assert m['CM_D_E_rows_D_E_U_columns']==[[1,0,1],[1,0,0]]and m['D_recall']==.5 and m['prediction_U_rate_known']==1/3
    assert decode(np.array([-1.,-0.,0.,1.,np.nan]),np.array([1,1,1,1,0],bool)).tolist()==[0,1,1,1,-1]
    generated=H/'GENERATED_SAVED_METRIC_CHECKS.json'
    write(generated,dict(completed=True,PASS=True,U_denominator_and_self_exclusion=True,hidden_zero_E_and_noquery_U=True,
                         code=bind(__file__),actual_model_RGB_checkpoint_or_private_reads=0))
    p['generated']=bind(generated);write(PROTOCOL,p)
    ep=H/'ROOT_REPORT_ENTRY_TEMPLATE.json'
    write(ep,dict(approved_saved_report=False,model_forward=False,fit=False,checkpoint_body_read=False,RGB_read=False,
                  current223=False,V112=False,PnP=False,label_edit=False,automatic_retry=False,code=p['code'],protocol=bind(PROTOCOL)))
    print(json.dumps(dict(completed=True,code=p['code'],protocol=bind(PROTOCOL),template=bind(ep),
                         command='/home/minjae/anaconda3/bin/python '+str(Path(__file__).resolve())+' --entry ROOT_REPORT_ENTRY.json',
                         actual_report_run=False)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare:prepare()
    elif a.entry:run(a.entry)
    else:raise SystemExit('Use --prepare or root --entry')
