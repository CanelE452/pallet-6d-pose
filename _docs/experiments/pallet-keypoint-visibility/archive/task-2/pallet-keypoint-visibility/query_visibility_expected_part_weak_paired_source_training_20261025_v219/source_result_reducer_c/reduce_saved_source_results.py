"""두 final state 뒤의 저장 SOURCE JSON만 축약한다. 입력/weights/이미지는 열지 않는다."""
import argparse, csv, hashlib, html, io, json, math, os, re, stat
from collections import Counter, defaultdict
from pathlib import Path
D=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_expected_part_weak_paired_source_training_20261025_v219')
H=D/'source_result_reducer_c';RUN=D/'paired_source_fit_v1';OUT=D/'saved_source_result_readout_v1'
WEAK=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_public_train_intended_part_query_targets_20261025_v218/joined_weak_intended_part_targets_v1/ALL360_CHECKED_WEAK_TARGET_CANDIDATES.jsonl')
WEAK_SHA='52ac5bb8d0fb19a1329b9bdb7f7b56964fed954a2a4d50489e3dcb7d598d139d'

def need(x,m):
    if not x:raise ValueError(m)

def read(path):
    p=Path(path);need(p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).lower(),'literal non-V112 path before filesystem')
    need(p.suffix in ('.json','.jsonl','.py'),'saved text only; never image/NPZ/weight')
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        leaf=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    with os.fdopen(leaf,'rb') as f:need(stat.S_ISREG(os.fstat(f.fileno()).st_mode),'regular text');return f.read()

def bind(p):
    z=read(p);return dict(path=str(p),sha256=hashlib.sha256(z).hexdigest(),size_bytes=len(z))

def load(b):
    z=read(b['path']);need(hashlib.sha256(z).hexdigest()==b['sha256'],'SHA binding');need('size_bytes' not in b or len(z)==b['size_bytes'],'optional size')
    return [json.loads(x) for x in z.splitlines() if x.strip()] if Path(b['path']).suffix=='.jsonl' else json.loads(z)

def save(name,obj):
    p=OUT/name
    with p.open('x') as f:f.write(obj if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    z=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(z).hexdigest(),size_bytes=len(z))

def decode(z):
    return -1 if z is None or isinstance(z,bool) or not isinstance(z,(float,int)) or not math.isfinite(z) else int(z>=0.)

def metric(rows):
    cm=[[0,0,0],[0,0,0]];unknown=0
    for r in rows:
        t=r['target'];p=r['prediction']
        if t not in (0,1):unknown+=1;continue
        cm[t][p if p in (0,1) else 2]+=1
    classes=[]
    for k in (0,1):
        truth=sum(cm[k]);pred=cm[0][k]+cm[1][k];tp=cm[k][k]
        classes.append(dict(class_D0_E1=k,truth=truth,predicted_on_known_truth=pred,correct=tp,precision=tp/pred if pred else None,recall=tp/truth if truth else None))
    return dict(rows=len(rows),truth_known=sum(map(sum,cm)),truth_UNKNOWN=unknown,CM_D_E_by_D_E_U=cm,classes=classes,model_U_on_known_truth=cm[0][2]+cm[1][2])

def pair(rows,control,treatment):
    out={}
    for truth in ['ALL',0,1]:
        rr=[r for r in rows if r['target'] in (0,1) and (truth=='ALL' or r['target']==truth)];cc=dd=rec=dam=0;ids_rec=[];ids_dam=[]
        for r in rr:
            a=r['predictions'][control]==r['target'];b=r['predictions'][treatment]==r['target']
            cc+=a and b;dd+=not a and not b;rec+=not a and b;dam+=a and not b
            if not a and b:ids_rec.append(r['row_id'])
            if a and not b:ids_dam.append(r['row_id'])
        out[str(truth)]=dict(rows=len(rr),both_correct=cc,both_wrong=dd,recovered_by_treatment=rec,damaged_by_treatment=dam,net_correct_change=rec-dam,recovered_row_ids=ids_rec,damaged_row_ids=ids_dam)
    return out

def strata(rows):
    out={}
    for key in ['native_slot','anatomical_family','group_id','component_id']:
        values={r.get(key,'UNKNOWN_METADATA') for r in rows}
        if key=='native_slot':values.update(range(8))
        if key=='anatomical_family':values.update(['DECK','BOTTOM_RUNNER'])
        out[key]={str(v):metric([r for r in rows if r.get(key,'UNKNOWN_METADATA')==v]) for v in sorted(values,key=str)}
    out['unique_groups']=len({r['group_id'] for r in rows}) if all(r.get('group_id') is not None for r in rows) else None
    out['unique_components']=len({r['component_id'] for r in rows}) if all(r.get('component_id') is not None for r in rows) else None
    return out

def reduce(entry_path):
    e=json.loads(read(entry_path));need(e['approved_saved_SOURCE_reduction'] is True,'separate root actual saved-only entry')
    need(e['code']==bind(__file__) and e['protocol']==bind(H/'PROTOCOL_BEFORE_FUTURE_SAVED_SOURCE_PREDICTIONS.json'),'exact reducer/protocol')
    scope=load(e['root_actual_scope']);need(scope['approved_saved_SOURCE_reduction'] is True,'root actual scope')
    need(Path(e['fit_complete']['path'])==RUN/'COMPLETE.json','only final declared source run COMPLETE')
    done=load(e['fit_complete']);need(done['completed'] is True and done['PASS'] is True and done['source_only'] is True and done['from_scratch'] is True,'completed fresh SOURCE run')
    control,treatment=e['control_arm'],e['treatment_arm'];need(control!=treatment and set(done['arms'])=={control,treatment},'retain both predeclared arms')
    need(done.get('selected_arm') is None,'no selected arm')
    freeze_b=done['all2_freeze'];need(Path(freeze_b['path']).is_relative_to(RUN),'own all2 barrier only');freeze=load(freeze_b)
    need(freeze['completed'] is True and freeze['PASS'] is True and freeze['DEV_or_real_output_emitted'] is False,'both final states before source DEV/real output')
    need(freeze['checkpoints']=={k:done['arms'][k]['checkpoint'] for k in (control,treatment)},'exact both checkpoint metadata closure; checkpoint bodies unopened')
    original=load(dict(path=str(WEAK),sha256=WEAK_SHA));need(len(original)==360 and len({r['row_id'] for r in original})==360,'all360 canonical target census')
    weakmeta={r['row_id']:r for r in original if r['weak_loss_candidate'] is True};need(len(weakmeta)==57,'57 support-checked weak targets;303 others separate')
    denominators={'SOURCE_FIT':{0:39,1:6},'SOURCE_DEV':{0:9,1:3}};nominal={'SOURCE_FIT':{0:15,1:5},'SOURCE_DEV':{0:4,1:3}}
    actual={a:{} for a in (control,treatment)};strong={};output_bindings={}
    for arm in actual:
        record=done['arms'][arm];output_bindings[arm]={k:record[k] for k in ['strong_outputs','weak57_outputs']}
        for b in output_bindings[arm].values():need(Path(b['path']).is_relative_to(RUN) and Path(b['path']).suffix=='.json','only own saved scalar JSON outputs')
        weakdata=load(record['weak57_outputs']);ww=weakdata['rows57'];need(len(ww)==57 and {r['row_id'] for r in ww}==set(weakmeta),'exact original weak row IDs')
        for r in ww:
            m=weakmeta[r['row_id']];g=m['generated_input'];target=m['weak_target_E_binary'];z=r['raw_logit'];pred=decode(z)
            need(r['target']==target and r['source_role']==m['source_role'] and target in (0,1),'no target/role relabel')
            need(pred==r['prediction_D0_E1'] and pred in (0,1),'fixed logit0 / finite known outputs modelU0')
            actual[arm][r['row_id']]=dict(row_id=r['row_id'],target=target,prediction=pred,raw_logit=z,source_role=m['source_role'],native_slot=g['native_slot'],anatomical_family=g['anatomical_family'],group_id=g['group_id'],component_id=g['component_id'],state_index=int(g['group_id'].rsplit('_S',1)[1]))
        ss=load(record['strong_outputs'])['rows'];need(len({r['row_id'] for r in ss})==len(ss),'strong row IDs unique')
        strongrows=[]
        for r in ss:
            pred=decode(r['raw_logit']);need(pred==r['prediction_D0_E1'] and pred in (0,1),'strong fixed decoder/finite modelU0')
            need(r['source_role']=='TRAIN' and r['target'] in (-1,0,1),'strong TRAIN apparent only')
            strongrows.append(dict(r,prediction=pred))
        need(Counter(r['target'] for r in strongrows if r['target']>=0)=={0:5853,1:243},'source synthetic eligible6096 classes')
        strong[arm]=dict(metrics=metric(strongrows),strata=strata(strongrows),full_source_point_census=7040,loss_eligible_known=6096,excluded_loss_rows=944,
            full7040_not_all_predicted=len(strongrows)!=7040,TRAIN_apparent_not_independent_validation=True,strata_missing_not_inferred_from_ID=True)
    combined=[]
    for rid,m in weakmeta.items():
        a=actual[control][rid];b=actual[treatment][rid];need({k:v for k,v in a.items() if k not in ['prediction','raw_logit']}=={k:v for k,v in b.items() if k not in ['prediction','raw_logit']},'paired exact metadata')
        combined.append(dict(a,predictions={control:a['prediction'],treatment:b['prediction']},raw_logits={control:a['raw_logit'],treatment:b['raw_logit']}))
    result={};csv_rows=[]
    for role in ['ALL','SOURCE_FIT','SOURCE_DEV']:
        for sub in ['ALL_STATES','NOMINAL_S0']:
            rr=[r for r in combined if (role=='ALL' or r['source_role']==role) and (sub=='ALL_STATES' or r['state_index']==0)]
            if role!='ALL':need(Counter(r['target'] for r in rr)==(denominators if sub=='ALL_STATES' else nominal)[role],'frozen role/subset truth denominators')
            key=role+'/'+sub;result[key]=dict(rows=len(rr),unique_group_ids=len({r['group_id'] for r in rr}),unique_components=len({r['component_id'] for r in rr}),unique_frames=len({weakmeta[r['row_id']]['generated_input']['frame_position'] for r in rr}),models={},paired=pair(rr,control,treatment))
            for arm in (control,treatment):
                ar=[dict(r,prediction=r['predictions'][arm]) for r in rr];mm=metric(ar);result[key]['models'][arm]=dict(metrics=mm,strata=strata(ar))
                for kind,values in [('ALL',{'ALL':mm})]+[(kind,result[key]['models'][arm]['strata'][kind]) for kind in ['native_slot','anatomical_family','group_id','component_id']]:
                    for value,sm in values.items():
                        for k in (0,1):csv_rows.append(dict(cohort=key,arm=arm,stratum_type=kind,stratum_value=value,rows=sm['rows'],**sm['classes'][k],model_U_on_known_truth=sm['model_U_on_known_truth'],CM=json.dumps(sm['CM_D_E_by_D_E_U'])))
    need(not OUT.is_symlink() and all(not p.is_symlink() for p in OUT.parents),'ordinary output path');OUT.mkdir(exist_ok=False)
    metrics=save('SOURCE_METRICS_AND_PAIRED.json',dict(completed=True,PASS=True,fit_complete=e['fit_complete'],all2_freeze=freeze_b,source_outputs=output_bindings,weak_canonical=dict(path=str(WEAK),sha256=WEAK_SHA),threshold_raw_logit=0.,models_retained=[control,treatment],weak=result,strong=strong,
        weak_actual_truth_known57=True,all360_target_census=360,all360_state_counts=dict(Counter(r['target_state'] for r in original)),unscored303_by_state=dict(Counter(r['target_state'] for r in original if r['weak_loss_candidate'] is not True)),other303_truth_location_unresolved_not_model_U=True,weak_DEV_preexposed_SOURCE_not_freshblind=True,assistant_consensus_not_human_or_native_point_gold=True,GT_derived_training_query_origin=True,no_protected_evaluation=True,no_arm_threshold_or_checkpoint_selection=True,adopted=False))
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows);csv_b=save('SOURCE_CONFUSIONS_PRECISION_RECALL.csv',buf.getvalue())
    paired_b=save('ALL57_PAIRED_SOURCE_ROWS.json',dict(completed=True,rows57=combined,all_original360_metadata_retained_in_canonical=True,other303_predictions_not_inferred=True))
    text='<html lang="ko"><meta charset="utf-8"><title>V219 SOURCE 저장 결과</title><style>body{font-family:sans-serif;max-width:1500px;margin:24px auto}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}img{image-rendering:pixelated;width:192px;height:192px}code{overflow-wrap:anywhere}</style><h1>두 팔의 저장 SOURCE 결과 · 전체 weak57</h1><p>고정 logit 0 기준. FIT45(D39/E6), DEV12(D9/E3). nominal S0는 FIT20(D15/E5), DEV7(D4/E3)입니다. 전체360 중 나머지303행의 truth/location 미지는 모델 U가 아닙니다. strong7040의 eligible6096은 TRAIN apparent이며 독립 검증이 아닙니다. DEV도 사전 노출 SOURCE로, fresh blind·human gold·native point gold가 아닙니다. 새 q8 자체는 GT-derived 학습 입력입니다. 승자·threshold·adoption을 고르지 않고 두 팔을 보존합니다.</p><p><a href="SOURCE_METRICS_AND_PAIRED.json">수치·strata</a> · <a href="SOURCE_CONFUSIONS_PRECISION_RECALL.csv">CSV</a> · <a href="ALL57_PAIRED_SOURCE_ROWS.json">57행 paired JSON</a></p><table><tr><th>원 RGB96 (새 이미지 수정 없음)</th><th>row / 역할 / 부위</th><th>truth</th><th>'+html.escape(control)+'</th><th>'+html.escape(treatment)+'</th></tr>'
    for r in combined:
        png=weakmeta[r['row_id']]['generated_input']['RGB96'];src=os.path.relpath(png['path'],OUT)
        text+='<tr><td><img alt="기존 SOURCE RGB96" src="'+html.escape(src,quote=True)+'"></td><td><code>'+html.escape(r['row_id'])+'</code><br>'+html.escape(r['source_role'])+' / n'+str(r['native_slot'])+' '+html.escape(r['anatomical_family'])+'</td><td>'+('E' if r['target'] else 'D')+'</td>'
        for arm in (control,treatment):text+='<td>logit '+format(r['raw_logits'][arm],'.7g')+'<br>prediction '+('E' if r['predictions'][arm] else 'D')+'</td>'
        text+='</tr>'
    text+='</table><p>기존57 PNG의 상대링크만 작성했습니다. reducer는 이미지 본문을 읽거나 수정하지 않았으며 브라우저 렌더를 수행하지 않았습니다.</p></html>'
    page=save('PORTABLE_SOURCE_READOUT_KO.html',text)
    complete=save('COMPLETE.json',dict(completed=True,PASS=True,entry=bind(entry_path),code=e['code'],protocol=e['protocol'],fit_complete=e['fit_complete'],all2_freeze=freeze_b,metrics=metrics,CSV=csv_b,paired_rows=paired_b,portable_HTML=page,image_weight_NPZ_GT_or_protected_reads=0,browser_rendered=False,selected_model=None))
    print(json.dumps(dict(COMPLETE=complete,weak_known57=True,fit45=True,DEV12=True),ensure_ascii=False))

def fixtures():
    need([decode(x) for x in [-1.,0.,1.,None,float('nan')]]==[0,1,1,-1,-1],'fixed decoder')
    r=[dict(target=0,prediction=0),dict(target=0,prediction=1),dict(target=1,prediction=-1),dict(target=-1,prediction=1)];m=metric(r);need(m['CM_D_E_by_D_E_U']==[[1,1,0],[0,0,1]] and m['truth_UNKNOWN']==1 and m['classes'][0]['precision']==1 and m['classes'][0]['recall']==.5 and m['classes'][1]['precision']==0 and m['classes'][1]['recall']==0,'CM/P/R unknown truth separate')
    p=pair([dict(row_id='D',target=0,predictions={'a':1,'b':0}),dict(row_id='E',target=1,predictions={'a':1,'b':0})],'a','b');need(p['ALL']['recovered_by_treatment']==p['ALL']['damaged_by_treatment']==1 and p['0']['net_correct_change']==1 and p['1']['net_correct_change']==-1,'paired recover/damage per truth')
    return dict(completed=True,PASS=True,groups3=['fixed0 decoder/nonfiniteU','CM precision/recall and truth_UNKNOWN separate','paired D/E recovery/damage'],actual_saved_prediction_or_image_reads=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry');p.add_argument('--invented-checks',action='store_true');a=p.parse_args()
    if a.invented_checks:print(json.dumps(fixtures()))
    else:need(a.entry is not None,'entry required');reduce(a.entry)
