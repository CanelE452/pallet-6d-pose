"""Independent Counter reduction of completed scalar SOURCE outputs; no producer import."""
import csv
import hashlib
import io
import json
import math
import os
import stat
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

D=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_expected_part_weak_paired_source_training_20261025_v219')
O=Path(__file__).parent
F=D/'paired_source_fit_v1/COMPLETE.json'
R=D/'saved_source_result_readout_v1/COMPLETE.json'
INPUTS={str(F):'e6c820488e111049056384bce482a2d1bfb9c9b964ea285da5093d8d0d941f8f',str(R):'d06c9489dc87422bd559c034d4aa81d942de185bd4b1e1bf63f1f50f08fb894a'}

def read(b):
    p=Path(b['path']);assert str(p) in INPUTS and INPUTS[str(p)]==b['sha256']
    assert p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).lower() and p.suffix in ('.json','.jsonl','.csv','.html','.py')
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        leaf=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    with os.fdopen(leaf,'rb') as f:
        assert stat.S_ISREG(os.fstat(f.fileno()).st_mode);data=f.read()
    assert hashlib.sha256(data).hexdigest()==b['sha256'] and ('size_bytes' not in b or len(data)==b['size_bytes'])
    return data.decode()

def add(b):
    p=Path(b['path']);assert p.suffix in ('.json','.jsonl','.csv','.html','.py') and 'v112' not in str(p).lower()
    if str(p) in INPUTS:assert INPUTS[str(p)]==b['sha256']
    INPUTS[str(p)]=b['sha256'];return b

def load(b):
    s=read(b)
    return [json.loads(line) for line in s.splitlines() if line.strip()] if b['path'].endswith('.jsonl') else json.loads(s)

def fixed(z):
    return int(z>=0) if type(z) in (int,float) and math.isfinite(z) else -1

def counts(rows):
    c=Counter((r['target'],r['pred']) for r in rows if r['target'] in (0,1))
    cm=[[c[t,p] for p in (0,1,-1)] for t in (0,1)]
    classes=[]
    for t in (0,1):
        truth=sum(cm[t]);pred=sum(cm[y][t] for y in (0,1));tp=cm[t][t]
        classes.append(dict(class_D0_E1=t,truth=truth,predicted_on_known_truth=pred,correct=tp,precision=tp/pred if pred else None,recall=tp/truth if truth else None))
    return dict(rows=len(rows),truth_known=sum(map(sum,cm)),truth_UNKNOWN=sum(r['target'] not in (0,1) for r in rows),CM_D_E_by_D_E_U=cm,classes=classes,model_U_on_known_truth=sum(cm[t][2] for t in (0,1)))

def equality(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a),set(b))
        for k in a:equality(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equality(x,y)
    elif type(a) is float:assert isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=0,abs_tol=1e-12),(a,b)
    else:assert a==b,(a,b)

def paired(rows,arms):
    answer={}
    for target in ('ALL',0,1):
        rr=[r for r in rows if target=='ALL' or r['target']==target]
        correct=Counter((r['preds'][arms[0]]==r['target'],r['preds'][arms[1]]==r['target']) for r in rr)
        recovered=[r['row_id'] for r in rr if r['preds'][arms[0]]!=r['target'] and r['preds'][arms[1]]==r['target']]
        damaged=[r['row_id'] for r in rr if r['preds'][arms[0]]==r['target'] and r['preds'][arms[1]]!=r['target']]
        answer[str(target)]=dict(rows=len(rr),both_correct=correct[True,True],both_wrong=correct[False,False],recovered_by_treatment=len(recovered),damaged_by_treatment=len(damaged),net_correct_change=len(recovered)-len(damaged),recovered_row_ids=recovered,damaged_row_ids=damaged)
    return answer

class Table(HTMLParser):
    def __init__(self):super().__init__();self.rows=[];self.row=None;self.cell=None;self.images=[];self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='tr':self.row=[]
        elif tag in ('td','th'):self.cell=''
        elif tag=='br' and self.cell is not None:self.cell+=' '
        elif tag=='img':self.images.append(a['src'])
        elif tag=='a':self.links.append(a['href'])
    def handle_data(self,data):
        if self.cell is not None:self.cell+=data
    def handle_endtag(self,tag):
        if tag in ('td','th') and self.cell is not None:self.row.append(' '.join(self.cell.split()));self.cell=None
        elif tag=='tr':self.rows.append(self.row);self.row=None

def main():
    fit=load(dict(path=str(F),sha256=INPUTS[str(F)]));done=load(dict(path=str(R),sha256=INPUTS[str(R)]))
    assert fit['completed'] and fit['PASS'] and fit['source_only'] and fit['from_scratch'] and fit['steps_each']==4000 and fit['parameters_each']==118359
    assert done['fit_complete']['sha256']==INPUTS[str(F)] and fit['selected_arm'] is None and not fit['adopted']
    arms=list(fit['arms']);assert arms==['STRONG_ONLY_CONTROL','STRONG_PLUS_EXPECTED_PART_WEAK']
    assert len({fit['arms'][a]['initial'] for a in arms})==len({fit['arms'][a]['old_point_draw_sha'] for a in arms})==1
    freeze=load(add(fit['all2_freeze']));assert freeze['checkpoints']=={a:fit['arms'][a]['checkpoint'] for a in arms} and freeze['DEV_or_real_output_emitted'] is False
    metrics=load(add(done['metrics']));protocol=load(add(done['protocol']));read(add(done['code']))
    canonical=load(add(metrics['weak_canonical']));assert len(canonical)==360 and len({r['row_id'] for r in canonical})==360
    known=[r for r in canonical if r['weak_loss_candidate'] is True];assert len(known)==57
    meta={r['row_id']:r for r in known};actual={};strong={}
    for arm in arms:
        actual[arm]={}
        for r in load(add(fit['arms'][arm]['weak57_outputs']))['rows57']:
            assert r['row_id'] not in actual[arm];m=meta[r['row_id']]
            assert r['target']==m['weak_target_E_binary'] and r['source_role']==m['source_role']
            p=fixed(r['raw_logit']);assert p==r['prediction_D0_E1'] and p in (0,1)
            actual[arm][r['row_id']]=r|{'pred':p}
        assert set(actual[arm])==set(meta)
        ss=load(add(fit['arms'][arm]['strong_outputs']))['rows'];assert len({r['row_id'] for r in ss})==len(ss)
        for r in ss:
            r['pred']=fixed(r['raw_logit']);assert r['pred']==r['prediction_D0_E1'] and r['pred'] in (0,1) and r['source_role']=='TRAIN'
        strong[arm]=ss;equality(counts(ss),metrics['strong'][arm]['metrics'])
        assert len(ss)==6096 and Counter(r['target'] for r in ss)=={0:5853,1:243}
        for key,values in metrics['strong'][arm]['strata'].items():
            if key in ('unique_groups','unique_components'):assert values is None
            else:
                for value,mm in values.items():equality(counts(ss if value=='UNKNOWN_METADATA' else []),mm)
    assert {r['row_id']:r['target'] for r in strong[arms[0]]}=={r['row_id']:r['target'] for r in strong[arms[1]]}
    combined=[]
    for m in known:
        g=m['generated_input'];rid=m['row_id']
        combined.append(dict(row_id=rid,target=m['weak_target_E_binary'],source_role=m['source_role'],native_slot=g['native_slot'],anatomical_family=g['anatomical_family'],group_id=g['group_id'],component_id=g['component_id'],frame_position=g['frame_position'],state_index=int(g['group_id'].rsplit('_S',1)[1]),preds={a:actual[a][rid]['pred'] for a in arms},logits={a:actual[a][rid]['raw_logit'] for a in arms}))
    paired_saved=load(add(done['paired_rows']))['rows57'];assert [r['row_id'] for r in paired_saved]==[r['row_id'] for r in combined]
    for r,s in zip(combined,paired_saved):
        for k in ('target','source_role','native_slot','anatomical_family','group_id','component_id','state_index'):assert r[k]==s[k]
        assert r['preds']==s['predictions'] and r['logits']==s['raw_logits']
    expected_csv={};headlines={};metric_cells=0
    for cohort,record in metrics['weak'].items():
        role,sub=cohort.split('/')
        rr=[r for r in combined if (role=='ALL' or r['source_role']==role) and (sub=='ALL_STATES' or r['state_index']==0)]
        assert record['rows']==len(rr) and record['unique_group_ids']==len({r['group_id'] for r in rr}) and record['unique_components']==len({r['component_id'] for r in rr}) and record['unique_frames']==len({r['frame_position'] for r in rr})
        equality(paired(rr,arms),record['paired'])
        headlines[cohort]={k:record[k] for k in ('rows','unique_group_ids','unique_components','unique_frames')}
        headlines[cohort]['models']={};headlines[cohort]['paired']=paired(rr,arms)
        for arm in arms:
            ar=[r|{'pred':r['preds'][arm]} for r in rr]
            mm=counts(ar);equality(mm,record['models'][arm]['metrics']);headlines[cohort]['models'][arm]=mm
            strata=record['models'][arm]['strata']
            assert strata['unique_groups']==len({r['group_id'] for r in ar}) and strata['unique_components']==len({r['component_id'] for r in ar})
            all_metrics=[('ALL','ALL',mm)]
            for key in ('native_slot','anatomical_family','group_id','component_id'):
                values=set(str(r[key]) for r in ar)
                if key=='native_slot':values|=set(map(str,range(8)))
                if key=='anatomical_family':values|={'DECK','BOTTOM_RUNNER'}
                assert set(strata[key])==values
                for value in values:
                    sm=counts([r for r in ar if str(r[key])==value]);equality(sm,strata[key][value]);all_metrics.append((key,value,sm));metric_cells+=1
            for kind,value,sm in all_metrics:
                for cls in sm['classes']:
                    expected_csv[cohort,arm,kind,value,str(cls['class_D0_E1'])]=dict(rows=sm['rows'],**cls,model_U_on_known_truth=sm['model_U_on_known_truth'],CM=sm['CM_D_E_by_D_E_U'])
    csv_rows=list(csv.DictReader(io.StringIO(read(add(done['CSV'])))));assert len(csv_rows)==len(expected_csv)
    seen=set()
    for r in csv_rows:
        key=tuple(r[k] for k in ('cohort','arm','stratum_type','stratum_value','class_D0_E1'));assert key not in seen;seen.add(key);expect=expected_csv[key]
        for k,v in expect.items():
            if k=='CM':assert json.loads(r[k])==v
            elif k in ('precision','recall'):assert r[k]=='' if v is None else math.isclose(float(r[k]),v,rel_tol=0,abs_tol=1e-12)
            else:assert int(r[k])==v
    assert metrics['all360_state_counts']==dict(Counter(r['target_state'] for r in canonical))
    assert metrics['unscored303_by_state']==dict(Counter(r['target_state'] for r in canonical if not r['weak_loss_candidate']))
    page=Table();page.feed(read(add(done['portable_HTML'])));assert len(page.rows)==58 and len(page.images)==57
    assert page.links==['SOURCE_METRICS_AND_PAIRED.json','SOURCE_CONFUSIONS_PRECISION_RECALL.csv','ALL57_PAIRED_SOURCE_ROWS.json']
    for row,img,r in zip(page.rows[1:],page.images,combined):
        assert r['row_id'] in row[1] and r['source_role'] in row[1] and row[2]==('E' if r['target'] else 'D')
        assert img==os.path.relpath(meta[r['row_id']]['generated_input']['RGB96']['path'],Path(done['portable_HTML']['path']).parent)
        for cell,arm in zip(row[3:],arms):assert format(r['logits'][arm],'.7g') in cell and ('prediction '+('E' if r['preds'][arm] else 'D')) in cell
    result=dict(completed=True,PASS=True,fit_complete=dict(path=str(F),sha256=INPUTS[str(F)]),reducer_complete=dict(path=str(R),sha256=INPUTS[str(R)]),all2_freeze=fit['all2_freeze'],independent_method='Counter confusion and correctness-pair tables; no producer import; scalar fixed0 decoder.',strong={a:counts(strong[a]) for a in arms},weak=headlines,weak_truth_unknown_unscored=303,strong_full_census=7040,strong_known_scored=6096,strong_excluded_not_scored=944,csv_rows_exact=len(csv_rows),weak_stratum_metrics_exact=metric_cells,HTML57_scalar_rows_and_source_paths_exact=True,HTML_browser_rendered=False,original_image_NPZ_PT_GT_or_protected_opens=0,input_bindings=[dict(path=p,sha256=s) for p,s in INPUTS.items()])
    out=O/'ACTUAL_SOURCE_NUMERICAL_QA.json'
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
    print(json.dumps(dict(PASS=True,QA=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),headlines={k:dict(N=v['rows'],components=v['unique_components'],CMs={a:m['CM_D_E_by_D_E_U'] for a,m in v['models'].items()},paired=v['paired']['ALL']) for k,v in headlines.items()}),ensure_ascii=False))

if __name__=='__main__':main()
