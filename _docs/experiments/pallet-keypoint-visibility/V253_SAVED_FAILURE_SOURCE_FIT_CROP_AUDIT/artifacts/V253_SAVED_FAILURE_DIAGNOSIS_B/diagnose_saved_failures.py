#!/usr/bin/env python3
"""V253 saved SOURCE scalar/cache diagnosis; no model, image or training APIs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np

ROOT_PROTOCOL_SHA='e9f2a2217930ce38fde59a8c1ebd922cd97f7ad8fe2db02cfd76dafa8511b513'
MODELS=('BASE','CONTROL','TREAT')
VIEWS=('full_CLS','fixed_centre_Q','centre_CLS')
CROP_GROUPS=('D_BASEcorrect_TREATfalseE','D_TREATcorrect','E_TREATcorrect','E_TREATmiss')
STAT_GROUPS=('ALL','D_TREATfalseE','D_BASEcorrect_TREATfalseE','D_BASEfalseE_TREATfalseE','D_TREATcorrect','E_TREATcorrect','E_TREATmiss')
SAMPLE_KEYS=('target','family','fit','available','row_id','condition')


def safe(value):
    p=Path(value)
    if not p.is_absolute() or '..' in p.parts or any('v112' in x.lower() or 'cache' in x.lower() or x.lower().endswith('.pyc') for x in p.parts):raise ValueError('Forbidden explicit path before FS')
    for x in reversed([p,*p.parents]):
        if x.is_symlink():raise ValueError('Symlink artifact/ancestor')
    return p


def binding(path):
    p=safe(path);h=hashlib.sha256();n=0
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b);n+=len(b)
    return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':n}


def checked(b):
    if set(b)!= {'path','sha256','size_bytes'} or binding(b['path'])!=b:raise ValueError('Exact input binding mismatch')
    return safe(b['path'])


def jread(p):return json.loads(safe(p).read_text(encoding='utf-8'))


def jwrite(p,j):
    p=safe(p);tmp=safe(str(p)+'.partial')
    if p.exists() or tmp.exists():raise ValueError('Fresh atomic leaf required')
    tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p);return binding(p)


def csvwrite(path,rows):
    with safe(path).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,separators=(',',':'),allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    return binding(path)


def validate_samples(s):
    if set(s)!=set(SAMPLE_KEYS):raise ValueError('Six declared sample members only')
    n=len(s['target'])
    for k,dtype in [('target',np.int8),('family',np.int8),('fit',bool),('available',bool)]:
        if s[k].dtype!=dtype or s[k].shape!=(n,):raise ValueError('Sample vector dtype/shape')
    if not np.isin(s['target'],[0,1]).all() or not np.isin(s['family'],[0,1]).all() or s['row_id'].shape!=(n,) or s['row_id'].dtype.kind!='U' or len(set(map(str,s['row_id'])))!=n:raise ValueError('Known D/E families and unique original row IDs')
    c=s['condition'];syn=s['family']==0
    if c.dtype!=np.float32 or c.shape!=(n,32) or not np.isfinite(c).all() or np.count_nonzero(c[~syn]):raise ValueError('Original condition32; YCB exactlyzero')
    h=c[syn,:8]
    if not np.isin(h,[0.,1.]).all() or not (h.sum(1)==1).all():raise ValueError('SYN original onehot input address')
    return s


def load_saved_rows(paths,samples):
    """Only named CSV paths supplied by caller. No absolute-source fallback."""
    n=len(samples['target']);out={}
    for name,key in zip(MODELS,('BASE_CSV','CONTROL_CSV','TREAT_CSV')):
        with safe(paths[key]).open(encoding='utf-8',newline='') as f:raw=list(csv.DictReader(f))
        if len(raw)!=n:raise ValueError('All original known rows retained')
        typed={}
        for r in raw:
            i=int(r['sample_index']);fam=int(r['family']);y=int(r['target'])
            if i in typed or not 0<=i<n or r['available'] not in ('True','False'):raise ValueError('Unique original sample index and boolean')
            av=r['available']=='True';role='FIT' if samples['fit'][i] else 'DEV'
            if r['row_id']!=str(samples['row_id'][i]) or fam!=int(samples['family'][i]) or y!=int(samples['target'][i]) or av!=bool(samples['available'][i]) or r['role']!=role:raise ValueError('Saved original role/label/identity/availability join')
            z=[float(r['logit0']),float(r['logit1'])];v=float(r['selected_family_E_logit']);prediction='U' if not av else ('E' if v>=0 else 'D')
            if not np.isfinite(z+[v]).all() or v!=z[fam] or r['prediction']!=prediction:raise ValueError('Fixed rawzero decode/finite scores')
            typed[i]={'sample_index':i,'row_id':r['row_id'],'family':fam,'role':role,'target':y,'available':av,'logit0':z[0],'logit1':z[1],'E_logit':v,'prediction':prediction}
        if set(typed)!=set(range(n)):raise ValueError('Original row order completeness')
        out[name]=[typed[i] for i in range(n)]
    return out


def validate_refs(generic,target,n):
    for r in (generic,target):
        if r.dtype!=np.float32 or r.shape!=(n,9) or not np.isfinite(r).all() or not np.isin(r[:,6:],[0.,1.]).all():raise ValueError('Frozen ref9 schema')
        for v in range(3):
            if np.count_nonzero(r[r[:,6+v]==0,v]) or np.count_nonzero(r[r[:,6+v]==0,3+v]):raise ValueError('Unsupported encoded pair zeros')
    if not np.array_equal(generic[:,:3],generic[:,3:6]):raise ValueError('Generic selected/next are intentional exactduplicates')


def make_diag_rows(saved,generic_ref9,target_ref9,samples):
    """Pure same-row join; original SYN input address, never canonical native gold."""
    validate_samples(samples);n=len(samples['target']);validate_refs(generic_ref9,target_ref9,n);out=[]
    for i in range(n):
        b=saved['BASE'][i];r={'sample_index':i,'row_id':b['row_id'],'family':b['family'],'role':b['role'],'target':b['target'],'truth':'E' if b['target'] else 'D','available':b['available'],'original_native_address':int(samples['condition'][i,:8].argmax()) if b['family']==0 else None,'native_authority':'ORIGINAL_INITIALIZER_ADDRESS_NOT_CANONICAL_GOLD' if b['family']==0 else 'UNKNOWN'}
        for name in MODELS:
            a=saved[name][i]
            for k in ('row_id','family','role','target','available'):
                if a[k]!=b[k]:raise ValueError('Three model rows must be identical apart from scores')
            r[name+'_prediction']=a['prediction'];r[name+'_E_logit']=a['E_logit'];r[name+'_logit0']=a['logit0'];r[name+'_logit1']=a['logit1']
        r['generic_ref9']=[float(v) for v in generic_ref9[i]];r['target_ref9']=[float(v) for v in target_ref9[i]]
        for prefix,ref in [('GENERIC',generic_ref9),('TARGET',target_ref9)]:
            for vi,view in enumerate(VIEWS):
                a=float(ref[i,vi]);b=float(ref[i,3+vi]);r[f'{prefix}_{view}_selectedcos']=a;r[f'{prefix}_{view}_nextcos']=b;r[f'{prefix}_{view}_margin']=a-b;r[f'{prefix}_{view}_supported']=bool(ref[i,6+vi])
        out.append(r)
    return out


def member(r,group):
    y=r['target'];b=r['BASE_prediction'];t=r['TREAT_prediction']
    return (group=='ALL' or (group=='D_TREATfalseE' and y==0 and t=='E') or (group=='D_BASEcorrect_TREATfalseE' and y==0 and b=='D' and t=='E') or (group=='D_BASEfalseE_TREATfalseE' and y==0 and b=='E' and t=='E') or (group=='D_TREATcorrect' and y==0 and t=='D') or (group=='E_TREATcorrect' and y==1 and t=='E') or (group=='E_TREATmiss' and y==1 and t!='E'))


def specs(include_role=True):
    for family in (0,1):
        for native in ([None,*range(8)] if family==0 else [None]):
            for role in (('FIT','DEV') if include_role else (None,)):
                for truth in ('ALL','D','E'):
                    for error in STAT_GROUPS:
                        yield {'family':family,'role':role,'original_native_address':native,'native_group':'UNKNOWN' if family==1 else ('ALL_ADDRESSES' if native is None else str(native)),'truth_filter':truth,'failure_group':error}


def takes(r,s):
    return r['family']==s['family'] and (s['role'] is None or r['role']==s['role']) and (s['original_native_address'] is None or r['original_native_address']==s['original_native_address']) and (s['truth_filter']=='ALL' or r['truth']==s['truth_filter']) and member(r,s['failure_group'])


def descriptives(values):
    a=np.asarray(values,np.float64)
    if not len(a):return {'count':0,'mean':None,'std':None,'min':None,'p10':None,'p25':None,'median':None,'p75':None,'p90':None,'max':None}
    if not np.isfinite(a).all():raise ValueError('Finite descriptive values')
    q=map(float,np.quantile(a,[0,.1,.25,.5,.75,.9,1]));return {'count':len(a),'mean':float(a.mean()),'std':float(a.std(ddof=0)),**dict(zip(('min','p10','p25','median','p75','p90','max'),q))}


def metrics(rows,name):
    cm=[[sum(r['target']==y and r[name+'_prediction']==p for r in rows) for p in ('D','E','U')] for y in (0,1)];d=sum(cm[0]);e=sum(cm[1]);pd=cm[0][0]+cm[1][0];pe=cm[0][1]+cm[1][1];rd=cm[0][0]/d if d else None;re=cm[1][1]/e if e else None
    return {'model':name,'rows':len(rows),'D':d,'E':e,'CM_D_E_by_D_E_U':cm,'precisionD':cm[0][0]/pd if pd else None,'precisionE':cm[1][1]/pe if pe else None,'recallD':rd,'recallE':re,'BR':(rd+re)/2 if rd is not None and re is not None else None,'U':cm[0][2]+cm[1][2]}


def transitions(rows,old,new):
    cm=[[sum(r[old+'_prediction']==a and r[new+'_prediction']==b for r in rows) for b in ('D','E','U')] for a in ('D','E','U')]
    oldok=[r[old+'_prediction']==r['truth'] for r in rows];newok=[r[new+'_prediction']==r['truth'] for r in rows]
    return {'old_model':old,'new_model':new,'rows':len(rows),'transition_D_E_U_by_D_E_U':cm,'correct_both':sum(a and b for a,b in zip(oldok,newok)),'old_correct_new_wrong':sum(a and not b for a,b in zip(oldok,newok)),'old_wrong_new_correct':sum(not a and b for a,b in zip(oldok,newok)),'wrong_both':sum(not a and not b for a,b in zip(oldok,newok)),'prediction_changes':sum(r[old+'_prediction']!=r[new+'_prediction'] for r in rows)}


def events(rows):
    d=[r for r in rows if r['target']==0];e=[r for r in rows if r['target']==1];total=sum(r['TREAT_prediction']=='E' for r in d);new=sum(r['BASE_prediction']=='D' and r['TREAT_prediction']=='E' for r in d);existing=sum(r['BASE_prediction']=='E' and r['TREAT_prediction']=='E' for r in d);other=sum(r['BASE_prediction']=='U' and r['TREAT_prediction']=='E' for r in d)
    if total!=new+existing+other:raise ValueError('FalseE partition identity')
    out={'rows':len(rows),'D':len(d),'E':len(e),'D_TREATfalseE_total':total,'D_BASEcorrect_to_TREATfalseE_new':new,'D_BASEfalseE_to_TREATfalseE_existing':existing,'D_BASEU_to_TREATfalseE':other,'D_TREATfalseE_rate_originalD':total/len(d) if d else None}
    for old in ('BASE','CONTROL'):
        out['E_'+old+'miss_to_TREATcorrect']=sum(r[old+'_prediction']!='E' and r['TREAT_prediction']=='E' for r in e);out['E_'+old+'correct_to_TREATmiss']=sum(r[old+'_prediction']=='E' and r['TREAT_prediction']!='E' for r in e)
    return out


def summarize(diag_rows):
    group_metrics=[];transition_rows=[];refstats=[];event_rows=[]
    for s in specs():
        rr=[r for r in diag_rows if takes(r,s)]
        for name in MODELS:group_metrics.append({**s,**metrics(rr,name)})
        for a,b in [('BASE','CONTROL'),('BASE','TREAT'),('CONTROL','TREAT')]:transition_rows.append({**s,**transitions(rr,a,b)})
        for view in VIEWS:
            for score in ('selectedcos','nextcos','margin'):
                gkey=f'GENERIC_{view}_{score}';tkey=f'TARGET_{view}_{score}'
                for reference in ('GENERIC','TARGET','TARGET_MINUS_GENERIC'):
                    prefix='GENERIC' if reference=='GENERIC' else 'TARGET';support=[r for r in rr if r[prefix+'_'+view+'_supported'] and (reference!='TARGET_MINUS_GENERIC' or r['GENERIC_'+view+'_supported'])]
                    values=[r[gkey] if reference=='GENERIC' else r[tkey] if reference=='TARGET' else r[tkey]-r[gkey] for r in support]
                    refstats.append({**s,'reference':reference,'view':view,'score':score,'original_rows':len(rr),'supported_rows':len(support),'unsupported_rows':len(rr)-len(support),**descriptives(values)})
        if s['truth_filter']=='ALL' and s['failure_group']=='ALL':event_rows.append({**s,**events(rr)})
    index={tuple(r[k] for k in ('family','original_native_address','truth_filter','failure_group','reference','view','score','role')):r for r in refstats};shifts=[]
    for r in refstats:
        if r['role']!='FIT':continue
        key=tuple(r[k] for k in ('family','original_native_address','truth_filter','failure_group','reference','view','score'))+('DEV',);d=index[key]
        shifts.append({**{k:r[k] for k in ('family','original_native_address','native_group','truth_filter','failure_group','reference','view','score')},'FIT_original_rows':r['original_rows'],'DEV_original_rows':d['original_rows'],'FIT_supported':r['count'],'DEV_supported':d['count'],'FIT_mean':r['mean'],'DEV_mean':d['mean'],'DEV_minus_FIT_mean':None if r['mean'] is None or d['mean'] is None else d['mean']-r['mean'],'FIT_median':r['median'],'DEV_median':d['median'],'DEV_minus_FIT_median':None if r['median'] is None or d['median'] is None else d['median']-r['median']})
    pooled={role:events([r for r in diag_rows if r['role']==role]) for role in ('FIT','DEV')}
    summary={'completed':True,'PASS':True,'counts':{'known':len(diag_rows),'FIT':sum(r['role']=='FIT' for r in diag_rows),'DEV':sum(r['role']=='DEV' for r in diag_rows),'original_rows':15040,'masked_S_U_not_E':7423},'fixed_D_falseE_and_E_change_events':pooled,'pooled_metrics':{name:{f'family{f}_{role}':metrics([r for r in diag_rows if r['family']==f and r['role']==role],name) for f in (0,1) for role in ('FIT','DEV')} for name in MODELS},'fixed_views':list(VIEWS),'joint_reference_vectors_all3views_retained_per_row':True,'statistic_quantiles':['min','p10','p25','median','p75','p90','max'],'no_threshold_view_or_model_selection':True,'descriptive_statistics_only_no_causal_claim':True,'current_admission':False}
    return {'group_metrics':group_metrics,'transitions':transition_rows,'reference_distributions':refstats,'FIT_DEV_shifts':shifts,'events':event_rows,'summary':summary}


def make_selection(diag_rows):
    cells=[];indices=set()
    for family in (0,1):
        for native in (range(8) if family==0 else (None,)):
            for group in CROP_GROUPS:
                eligible=sorted((r for r in diag_rows if r['role']=='FIT' and r['family']==family and (native is None or r['original_native_address']==native) and member(r,group)),key=lambda r:r['sample_index']);chosen=eligible[:2]
                cells.append({'family':family,'original_native_address':native,'failure_group':group,'eligible_count':len(eligible),'selected':chosen,'empty':not bool(chosen)});indices.update(r['sample_index'] for r in chosen)
    if len(cells)!=36 or len(indices)>72 or any(r['role']!='FIT' for c in cells for r in c['selected']):raise ValueError('36FITcells/max72 deterministic selection')
    return {'completed':True,'PASS':True,'source_role':'FIT','rule':'FIRST2_ASCENDING_SAMPLE_INDEX_PER_PREDECLARED_CELL','examples_per_cell':2,'SYN_native_addresses':list(range(8)),'YCB_native':'UNKNOWN','groups':cells,'selected_sample_indices':sorted(indices),'unique_selected_rows':len(indices),'empty_cells':sum(c['empty'] for c in cells),'total_cells':36,'no_pixels_or_new_labels_read_or_created':True}


def write_results(output,diag_rows,tables,selection):
    out=safe(output)
    if out.exists():raise ValueError('Fresh diagnostic output required')
    out.mkdir(parents=False)
    names={'ALL7617_SAVED_PREDICTION_AND_REFERENCE_JOIN.csv':diag_rows,'GROUP_METRICS.csv':tables['group_metrics'],'TRANSITION_COUNTS.csv':tables['transitions'],'REFERENCE_DISTRIBUTIONS.csv':tables['reference_distributions'],'FIT_DEV_REFERENCE_SHIFTS.csv':tables['FIT_DEV_shifts'],'FIXED_D_DAMAGE_COUNTS.csv':tables['events']}
    outputs={name:csvwrite(out/name,rows) for name,rows in names.items()}
    outputs['FIT_TILE_SELECTION.json']=jwrite(out/'FIT_TILE_SELECTION.json',selection);outputs['SUMMARY.json']=jwrite(out/'SUMMARY.json',tables['summary'])
    message='# V253 SOURCE 저장 실패·reference 분포 진단\n\n이 결과는 저장된 세 모델 점수와 두 reference9 cache만의 기술적 진단입니다. 새 모델 forward·학습·bank 적합·문턱 조정·픽셀 열람은 없습니다.\n\n'+json.dumps(tables['summary']['fixed_D_falseE_and_E_change_events'],ensure_ascii=False,indent=2)+'\n\ntrueD가 TREAT E가 된 전체 수와 BASE D정답에서 새로 손상된 수를 구분합니다. 원 분모를 유지한 모든 가족/FIT·DEV/D·E/원SYN주소/세 view 표와 빈 셀을 보존했습니다. score 분포차이는 원인·라벨 오류·정보 불가능성의 증명이 아닙니다. FIT-D reference 자기포함과 반복 노출된 SOURCE DEV의 한계를 유지합니다. SYN 원native 입력주소는 physical canonical gold가 아니며 YCB associated-r 보조 라벨의 native·SELF는 UNKNOWN입니다.\n\nFIT_TILE_SELECTION은 36개 사전 고정 셀에서 sample index 최소 두 개와 빈 셀을 보존한 시각화 후보일 뿐, 픽셀 판독·새 라벨·학습 admission이 아닙니다. GT/r/owner를 추가하지 않았으며 현재 실사 개선이나 채택을 주장하지 않습니다.\n'
    safe(out/'PORTABLE_REPORT_KO.md').write_text(message,encoding='utf-8');outputs['PORTABLE_REPORT_KO.md']=binding(out/'PORTABLE_REPORT_KO.md')
    return outputs


def authorize(entry_path):
    e=jread(entry_path)
    if e.get('approved_saved_SOURCE_diagnosis') is not True or e.get('automatic_retry') is not False:raise PermissionError('Root true savedSOURCE entry required before payloads')
    if checked(e['code'])!=safe(__file__):raise ValueError('Selfcode binding')
    p=jread(checked(e['root_protocol']))
    if e['root_protocol']['sha256']!=ROOT_PROTOCOL_SHA:raise ValueError('Frozen Root protocol')
    keys=(*p['diagnostic_numeric_inputs'],'fit_complete');expected={k:p['inputs'][k] for k in keys}
    if e['inputs']!=expected or len(expected)!=7:raise ValueError('Exact6numeric+fitCOMPLETE only; no visualpayloads')
    checked(e['implementation_protocol']);scope=jread(checked(e['root_scope']))
    if scope.get('approved') is not True or scope.get('entry_contract')!={k:v for k,v in e.items() if k!='root_scope'}:raise ValueError('Exact Root scope')
    for b in e['inputs'].values():checked(b)
    old=jread(e['inputs']['fit_complete']['path'])
    if old.get('completed') is not True or old.get('PASS') is not True:raise ValueError('Completed saved fit')
    aliases={'BASE_CSV':old['BASE_FROZEN_V239_CSV'],'CONTROL_CSV':old['outputs']['CONTROL_GENERIC_REFERENCE'],'TREAT_CSV':old['outputs']['TREAT_TARGET_REFERENCE'],'generic_ref9':old['generic_reference_inputs'],'target_ref9':old['target_reference_inputs'],'samples':old['inputs']['samples']}
    if aliases!={k:e['inputs'][k] for k in p['diagnostic_numeric_inputs']}:raise ValueError('Saved fit identity closure')
    output=safe(e['output_directory']);want=Path(__file__).absolute().parent.parent/'V253_ROOT_SAVED_FAILURE_DIAGNOSIS/saved_failure_diagnosis_v1'
    if output!=want or output.exists():raise ValueError('Exact fresh Root output')
    return e,output


def run(entry_path):
    e,out=authorize(entry_path);p={k:safe(v['path']) for k,v in e['inputs'].items()}
    with np.load(p['samples'],allow_pickle=False) as z:s=validate_samples({k:z[k] for k in SAMPLE_KEYS})
    if len(s['target'])!=7617 or int(s['fit'].sum())!=5876 or int((~s['fit']).sum())!=1741:raise ValueError('Original7617/FIT5876/DEV1741')
    generic=np.load(p['generic_ref9'],mmap_mode='r',allow_pickle=False);target=np.load(p['target_ref9'],mmap_mode='r',allow_pickle=False)
    saved=load_saved_rows(p,s);rows=make_diag_rows(saved,generic,target,s);tables=summarize(rows);selection=make_selection(rows);outputs=write_results(out,rows,tables,selection)
    for b in e['inputs'].values():checked(b)
    cp=jwrite(out/'COMPLETE.json',{'completed':True,'PASS':True,'entry':binding(entry_path),'code':e['code'],'root_protocol':e['root_protocol'],'implementation_protocol':e['implementation_protocol'],'inputs':e['inputs'],'outputs':outputs,'counts':tables['summary']['counts'],'selected_FIT_rows':selection['unique_selected_rows'],'selection_cells':36,'selection_empty_cells':selection['empty_cells'],'new_forward_training_CP_RGB_GT_or_protected_reads':0,'source_teacher_derived_labels_and_condition_read':True,'existing_SOURCE_REF_input_scores_read':True,'descriptive_statistics_computed':True,'no_threshold_or_model_or_view_choice':True,'current_admission':False,'automatic_retry':False})
    print(json.dumps({'completed':True,'PASS':True,'COMPLETE':cp,'selected_FIT_rows':selection['unique_selected_rows']}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);a=p.parse_args();run(a.entry)


if __name__=='__main__':main()
