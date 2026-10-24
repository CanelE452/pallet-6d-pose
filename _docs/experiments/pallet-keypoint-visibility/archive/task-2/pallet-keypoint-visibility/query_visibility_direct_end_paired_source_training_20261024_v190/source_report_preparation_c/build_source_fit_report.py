"""Root-entry-only SOURCE TRAIN report. Preparation reads metadata/toys only.

No NumPy, torch, arrays, checkpoints, images, evaluation, network, or Git loader.
All16000 losses are summarized only after the separately approved report entry.
"""
from pathlib import Path
import argparse, hashlib, html, json, math, os, stat

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SOURCE_ONLY_FINAL_REPORT_AND_PUBLICATION_PREPARATION_SCOPE.json'
PROTOCOL=H/'PROTOCOL_BEFORE_TRAIN_LOG_OR_METRIC_BODIES.json'
ARMS=('C54_RAW96_DIRECT_END','C55_RAW96_PLUS_FROZEN_SAM_DIRECT_END')
DOMAINS={'SYNTH_AUG65536':65536,'PUBLIC_TRAIN1480':1480,'CF3816':3816,'DIRECT_CONTROL':2792}
EXPECTED_U={'SYNTH_AUG65536':1264,'PUBLIC_TRAIN1480':194,'CF3816':360,'DIRECT_CONTROL':60}
META=(
 ('runs/C54_C55_DIRECT_END_SOURCE_FIXED8000/COMPLETE.json','193116d6e7a82db499764febfee1ed2ff684cd6881e0dbbfa6c605943341ecc9'),
 ('independent_source_fit_review_b/actual_review_v1/REVIEW.json','3bd43d7274393adaea87014a8edcbc040d99f03383a88dde3bacd3534682d4a3'),
 ('direct_control_raw_sam_cache_v1/COMPLETE.json','ada310cc1134fa8148b7dcda3e8aeeb3e42b2b2dde6f18db992fd65ff705307d'))

def need(value,message):
    if not bool(value):raise AssertionError(message)
def safe(path):
    p=Path(path);need(p.is_absolute() and '..' not in p.parts and all('v112' not in s.casefold() for s in p.parts),'Lexical path guard before filesystem')
    need(P in p.parents,'Exact project descendant only');return p
def read_bytes(path):
    p=safe(path);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for name in p.parts[1:-1]:
            nxt=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        last=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(last).st_mode),'Regular text only')
    with os.fdopen(last,'rb') as f:return f.read()
def digest(data):return hashlib.sha256(data).hexdigest()
def bind(path):return {'path':str(path),'sha256':digest(read_bytes(path))}
def checked(binding):
    data=read_bytes(binding['path']);need(digest(data)==binding['sha256'],'Bound text SHA drift');return data
def obj(binding):return json.loads(checked(binding))
def write(path,value):
    data=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with path.open('x',encoding='utf-8') as f:f.write(data)
def loss_summary(rows,expected):
    keys=('old_five_mean','direct_loss','total_loss','grad_norm')
    values={k:[] for k in keys};components=[[] for _ in range(5)]
    for step,r in enumerate(rows,1):
        need(type(r['update']) is int and r['update']==step,'Exact consecutive loss rows')
        need(len(r['old_five_losses'])==5,'All five original component means')
        for k in keys:
            value=r[k];need(type(value) in (int,float) and math.isfinite(value),'Finite logged '+k);values[k].append(float(value))
        for j,value in enumerate(r['old_five_losses']):
            need(type(value) in (int,float) and math.isfinite(value),'Finite old component');components[j].append(float(value))
    need(len(values['total_loss'])==expected,'Full logged update denominator')
    def stats(v):return dict(rows=len(v),first=v[0],last=v[-1],minimum=min(v),maximum=max(v),mean=math.fsum(v)/len(v))
    return dict(updates=expected,all_logged_rows_included=True,losses={k:stats(v) for k,v in values.items()},old_five_losses=[stats(v) for v in components])
def validate_direct(block,expected_rows=None,expected_truth=None):
    rows=block['rows'];known=block['known'];unknown=block['truth_UNKNOWN'];cm=block['CM_D_E_by_D_E_U']
    need(all(type(v) is int and v>=0 for v in [rows,known,unknown]),'Integer full denominators')
    need(rows==known+unknown and (expected_rows is None or rows==expected_rows),'UNKNOWN retained separately')
    need(len(cm)==2 and all(len(r)==3 for r in cm) and all(type(v) is int and v>=0 for r in cm for v in r),'D/E by D/E/U CM')
    need(sum(map(sum,cm))==known,'Known-truth CM denominator only')
    if expected_truth is not None:need([sum(r) for r in cm]==expected_truth,'Fixed direct source D/E class counts')
    need(block['known_abstentions']==sum(r[2] for r in cm),'Known abstentions denominator')
    return True
def rate(value):return '—' if value is None else f'{100*value:.2f}%'
def metric_rows(metrics):
    result=[]
    for arm in ARMS:
        block=metrics['models'][arm]['source_outputs']['DIRECT_CONTROL']['TRAIN_APPARENT_DIRECT_ONLY']['ALL']
        validate_direct(block,2792,[204,102]);cm=block['CM_D_E_by_D_E_U']
        for k in range(2):
            truth=sum(cm[k]);predicted=sum(r[k] for r in cm);correct=cm[k][k]
            result.append([arm,'보이는 끝 D' if k==0 else '외부 차단 끝 E',str(truth),str(correct),rate(correct/predicted if predicted else None),rate(correct/truth if truth else None),str(cm[k][2])])
    return result
def markdown_table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(r)+' |\n' for r in rows)
def make_report(fit,review,cache,metrics,losses,bindings):
    need(metrics['completed'] is True and metrics['PASS'] is True and metrics['not_validation_or_real_improvement'] is True,'Audited SOURCE TRAIN apparent only')
    need(metrics['known306']==306 and metrics['truthUNKNOWN2486']==2486 and set(metrics['models'])==set(ARMS),'Fixed full source direct census')
    for arm in ARMS:
        outputs=metrics['models'][arm]['source_outputs'];need(set(outputs)==set(DOMAINS),'All four source domains retained')
        for domain,n in DOMAINS.items():
            d=outputs[domain];need(d['rows']==n and d['inputU']==EXPECTED_U[domain],'Full source and inputU denominators')
            need(set(d['direct_predictions'])=={'0','1','-1'} and sum(d['direct_predictions'].values())==n,'All direct predictions inclU')
        direct=outputs['DIRECT_CONTROL']['TRAIN_APPARENT_DIRECT_ONLY']
        need(all('native:'+str(k) in direct for k in range(8)),'All eight native strata preserved')
        for group,b in direct.items():validate_direct(b)
    table=markdown_table(['모델','학습 표적','정답 분모','맞음','Precision','Recall','보류 U'],metric_rows(metrics))
    lossrows=[]
    for arm in ARMS:
        for key in ('old_five_mean','direct_loss','total_loss'):
            z=losses[arm]['losses'][key];lossrows.append([arm,key,str(z['rows']),f"{z['first']:.6g}",f"{z['last']:.6g}",f"{z['mean']:.6g}"])
    losstable=markdown_table(['모델','로그 항목','전체 step 수','첫 값','마지막 값','전체 평균'],lossrows)
    source='../query_visibility_sampled_domain_direct_end_20261024_v189/'
    links=[('원본 source 통제·실패·UNKNOWN 기록',source+'PORTABLE_REPORT_KO.md'),('합성 source 전체 gallery',source+'synthetic_end_review_visuals_b/PORTABLE_GALLERY_KO.md'),('실사 weak source gallery',source+'independent_actual_real_review_a/actual_saved_real_review_v1/PORTABLE_GALLERY_KO.md'),('학습 완료 원기록','runs/C54_C55_DIRECT_END_SOURCE_FIXED8000/COMPLETE.json'),('독립 학습 감사','independent_source_fit_review_b/actual_review_v1/REVIEW.json'),('TRAIN apparent 원수치','independent_source_fit_review_b/actual_review_v1/SOURCE_TRAIN_APPARENT_METRICS.json')]
    text=('# V190 source 직접 끝 표적 학습 완료\n\n'
      'RAW C54와 RAW+동결 SAM C55를 같은 초기화와 같은 표본 순서로 각각 8,000회 학습했습니다. 두 최종 가중치를 먼저 동결한 뒤 source 저장 출력을 만들었고 독립 감사가 통과했습니다.\n\n'
      '아래 수치는 학습에 사용한 SOURCE TRAIN 306개 직접 표적의 적합 결과입니다. 실사 전이·독립 검증·사용자 실사 성능의 개선 증거가 아닙니다. 2,792행 중 표적 UNKNOWN 2,486행과 입력 U 60행은 별도 분모로 유지했습니다. D 204/E 102는 합성 finite-end 구성 증거와 실사 assistant weak 의미 관찰을 포함하며 물리 native 정답으로 승격하지 않습니다.\n\n'+table+'\n'
      '기존 visible/foreign 두 support 손실과 owner 손실은 그대로 유지하고, 별도 direct-end logit 65개 파라미터 및 balanced 16D+16E 손실을 가중치 1로 더했습니다. 중심 owner FOREIGN, 넓은 support, native 가시성, 좌표 품질을 직접 끝 표적으로 치환하지 않았습니다.\n\n'
      '전체 16,000 step 로그를 빠짐없이 요약했습니다. 손실 크기나 감소는 실제 장면 일반화의 증거가 아닙니다.\n\n'+losstable+'\n'
      '기존 합성 증강 65,536행·공개 TRAIN 1,480행·CF 3,816행과 source 직접 통제 2,792행을 모두 보존했습니다. 공개/CF의 기존 owner 정답은 owner 보조 손실 전용이며 새 direct-end 정답으로 바꾸지 않았습니다.\n\n'
      '이 단계는 source 학습·기록의 완료입니다. 실사 평가와 비교는 별도 단계이며 이 보고서에 포함하지 않습니다. 모델 채택은 없고 전체 가림 목표는 미해결입니다.\n\n'+''.join('- ['+a+']('+b+')\n' for a,b in links)+'\n[전체 수치·authority·native strata](ROOT_RESULTS_KO.json)\n')
    results=dict(completed=True,PASS=True,phase='V190',stage='SOURCE_TRAIN_FIT_AND_APPARENT_READOUT',fit_completed=True,updates_each=8000,total_logged_updates=16000,known_direct306=306,direct_truth_UNKNOWN2486=2486,direct_rows2792=2792,direct_inputU60=60,direct_truth_classes_D_E=[204,102],source_domains=DOMAINS,source_inputU=EXPECTED_U,training_apparent_not_real_validation=True,losses=losses,source_train_apparent_metrics=metrics,bindings=bindings,links=[dict(label=a,path=b) for a,b in links],source_weak_authorities=cache['distinct_target_authorities'],current_publicheld_VAL_or_COMMON_evaluation_in_this_report=False,selected_model=None,adopted=False,fully_solved=False)
    return results,text
def run(entry_path):
    e=json.loads(read_bytes(entry_path));need(e['approved_source_train_report'] is True and e['no_evaluation_or_model_access'] is True,'Separate root report entry required')
    protocol=obj(e['protocol']);need(e['code']==bind(__file__) and protocol['code']==e['code'],'Frozen report source')
    need(e['root_scope']==protocol['root_scope'],'Exact prospective report scope');obj(e['root_scope'])
    expected=[dict(path=str(V/p),sha256=s) for p,s in META];need(protocol['metadata']==expected,'Exact three allowed metadata bindings')
    fit,review,cache=[obj(b) for b in expected]
    need(all(x['completed'] is True and x['PASS'] is True for x in [fit,review,cache]),'All three completed metadata PASS')
    need(fit['updates_each']==8000 and fit['total_updates']==16000 and fit['all2_frozen_before_outputs'] is True,'Both final8000 freeze barrier')
    need(set(fit['arms'])==set(ARMS) and fit['evaluation_or_COMMON_or_publicheld_or_private_access']==0,'Fixed source-only arms')
    need(cache['rows']==2792 and cache['original_inputU']==60 and cache['direct_known_counts']==[204,102],'Full admitted source census')
    need(review['all16000_log_rows_verified'] is True and review['all_old192_and_direct32_RNG_exact'] is True and review['direct_known306_unknown2486_separate'] is True,'Independent complete source audit')
    metrics=obj(review['metrics']);losses={};logs={}
    for arm in ARMS:
        a=fit['arms'][arm];need(a['updates']==8000 and a['initial_state_sha']==fit['same_initial_state_sha'] and a['old_draw_sha']==fit['same_old192_draw_sha'] and a['direct_draw_sha']==fit['same_direct32_draw_sha'],'Both arms initialization/draw parity')
        b=a['draws_and_losses'];need(Path(b['path']).parent==Path(expected[0]['path']).parent/arm and Path(b['path']).name=='ALL8000_DRAWS_AND_LOSSES.jsonl','Only declared source loss logs')
        data=checked(b);losses[arm]=loss_summary((json.loads(s) for s in data.splitlines()),8000);logs[arm]=b
    bindings=dict(metadata=expected,metrics=review['metrics'],all16000_logs=logs,code=e['code'],protocol=e['protocol'],root_entry=bind(entry_path))
    results,md=make_report(fit,review,cache,metrics,losses,bindings)
    for name in ['ROOT_RESULTS_KO.json','ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html','ROOT_PHASE_RESULT.json','ROOT_SOURCE_REPORT_BUILD_COMPLETE.json']:need(not (V/name).exists(),'No report overwrite')
    write(V/'ROOT_RESULTS_KO.json',results);write(V/'ROOT_REPORT_KO.md',md);write(V/'PORTABLE_REPORT_KO.md',md)
    # Small static HTML, source only; no browser execution/render assertion.
    body='<html lang="ko"><meta charset="utf-8"><title>V190 source 학습 기록</title><style>body{max-width:1100px;margin:30px auto;font:16px sans-serif;line-height:1.65}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:7px}pre{white-space:pre-wrap}</style><h1>V190 source 학습 기록</h1><p>각 8,000회 학습 완료. 아래는 TRAIN apparent 적합 결과이며 실사 일반화 검증이 아닙니다. UNKNOWN 2,486/2,792, 입력 U60을 별도로 보존했습니다. 채택 없음 · 전체 목표 미해결.</p>'
    body+='<table><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in ['모델','학습 표적','분모','맞음','Precision','Recall','U'])+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in r)+'</tr>' for r in metric_rows(metrics))+'</table><p>별도 direct-end 표적은 owner/넓은 support/native/quality와 다릅니다. 학습 손실·source 적합은 현재 실사 성능 개선 증거가 아닙니다.</p><p><a href="PORTABLE_REPORT_KO.md">상세 보고서·전체 로그 요약·source gallery</a> · <a href="ROOT_RESULTS_KO.json">전체 수치</a></p><details><summary>원문 보고서</summary><pre>'+html.escape(md)+'</pre></details></html>'
    write(V/'ROOT_READOUT_KO.html',body)
    phase=dict(completed=True,PASS=True,phase='V190',phase_complete_scope='Fixed source-only two-head fit and TRAIN apparent report only',results=bind(V/'ROOT_RESULTS_KO.json'),readout=bind(V/'ROOT_READOUT_KO.html'),portable=bind(V/'PORTABLE_REPORT_KO.md'),source_fit=expected[0],independent_source_fit_review=expected[1],source_input_cache=expected[2],report_generation_no_models_arrays_images_or_eval=True,selected_model=None,adopted=False,fully_solved=False,goal_status='active_unsolved',next_work='Separately frozen real evaluation; its status and outcomes are not included here.')
    write(V/'ROOT_PHASE_RESULT.json',phase);write(V/'ROOT_SOURCE_REPORT_BUILD_COMPLETE.json',dict(completed=True,PASS=True,phase=bind(V/'ROOT_PHASE_RESULT.json'),code=e['code'],protocol=e['protocol'],actual_entry=bind(entry_path),all16000_losses_summarized=True,model_array_image_evaluation_reads=0,HTML_browser_render_verified=False));print('SOURCE_TRAIN_REPORT_COMPLETE',flush=True)
def prepare():
    checks=[]
    rows=[dict(update=i,old_five_losses=[i]*5,old_five_mean=i,direct_loss=i/2,total_loss=i*1.5,grad_norm=i/4) for i in [1,2,3]]
    z=loss_summary(rows,3);checks.append(('all_rows_loss_summary',z['losses']['total_loss']['mean']==3 and z['updates']==3))
    for name,bad in [('missing_update',rows[:2]),('swapped_update',[rows[1],rows[0],rows[2]]),('nonfinite_loss',[dict(rows[0],total_loss=float('nan'))]+rows[1:])]:
        try:loss_summary(bad,3);ok=False
        except AssertionError:ok=True
        checks.append((name+'_rejected',ok))
    cm=dict(rows=8,known=5,truth_UNKNOWN=3,CM_D_E_by_D_E_U=[[2,0,1],[1,1,0]],known_abstentions=1)
    checks.append(('unknown_not_hidden_in_CM',validate_direct(cm,8,[3,2])))
    try:validate_direct(dict(cm,truth_UNKNOWN=0));ok=False
    except AssertionError:ok=True
    checks.append(('unknown_denominator_drift_rejected',ok))
    try:safe(P/'forbidden_v112'/'x');ok=False
    except AssertionError:ok=True
    checks.append(('forbidden_name_guard_before_read',ok))
    receipt=dict(completed=True,PASS=all(x for _,x in checks),checks=[dict(name=n,PASS=v) for n,v in checks],invented_only=True,actual_TRAIN_log_or_metric_model_array_image_eval_reads=0)
    write(H/'GENERATED_INVENTED_REPORT_CHECKS.json',receipt)
    protocol=dict(completed=True,prepared_only=True,code=bind(__file__),root_scope=dict(path=str(SCOPE),sha256='43bf1cbda9b3dcb539eb1d64f14610d20e51a8b2bd45c831180b38ac586274f7'),metadata=[dict(path=str(V/p),sha256=s) for p,s in META],actual_inputs='Only three bound COMPLETE/REVIEW JSON; metrics via REVIEW.metrics, and two ALL8000_DRAWS_AND_LOSSES JSONL via fit.arms. No source arrays/images/checkpoints/evaluation. Read only after separate root report entry.',frozen_report='Both8000/sameinit/draws/sourceonly and306known/2486UNKNOWN/2792rows/U60 exact; all16000 finite loss rows summarized, no step/model/threshold selection; source weak authorities remain distinct.',outputs=['ROOT_RESULTS_KO.json','ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html','ROOT_PHASE_RESULT.json','ROOT_SOURCE_REPORT_BUILD_COMPLETE.json'],portable_links='Reuse immutable V189 relative source galleries only. No new images/browser claim.',claims='TRAIN apparent is not validation/real improvement; current/publicheld/VAL/COMMON absent; no adoption, full goal unsolved. Report phase completion covers source fit only.',retry='Exclusive output; no automatic retry.')
    write(PROTOCOL,protocol);write(H/'ROOT_ACTUAL_SOURCE_REPORT_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_source_train_report=False,no_evaluation_or_model_access=True,code=protocol['code'],protocol=bind(PROTOCOL),root_scope=protocol['root_scope']))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=receipt['PASS'],code=protocol['code'],protocol=bind(PROTOCOL),checks=bind(H/'GENERATED_INVENTED_REPORT_CHECKS.json'),false_entry=bind(H/'ROOT_ACTUAL_SOURCE_REPORT_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_TRAIN_log_metric_array_model_eval_reads=0));print(json.dumps(receipt),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--entry',type=Path);a=p.parse_args()
    if a.prepare:prepare()
    elif a.entry:run(a.entry)
    else:p.error('--prepare or root --entry required')
