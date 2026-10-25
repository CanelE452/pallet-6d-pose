"""V233 저장 SOURCE 출력만 검산한다. 학습/모델/입력 RGB/GT 재조회 없음."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H = R / 'V233_SOURCE_REPORT_C'
CROP = R / 'V233_ROOT_CROP_ADAPTATION/crop_samples_v1'
PROTOCOL = dict(path=str(R/'V233_ROOT_CROP_ADAPTATION/PROTOCOL_BEFORE_SOURCE_PACKET_RGB_WEIGHT_FIT.json'),sha256='6755fecc3fd7927163357eb0caa677f4635e135f7819c954ae52316f3f29b28b')
TRAIN_CODE = dict(path=str(R/'V233_CROP_ADAPT_TRAINER_A/train_crop_adaptation.py'),sha256='3ca5e01dc27293cbc3da9df25b0411ea9286593c82a93aa0fb162c50aea0226a')
TRAIN_ENTRY = dict(path=str(R/'V233_ROOT_CROP_ADAPTATION/ACTUAL_PAIRED_FIT_ENTRY.json'),sha256='6d4ea7cf4a6d84ea6fa90b345108b9df7b864b8c3d5a30e638a16b93b61d02c5')
ARMS = ('CONTROL_LOCAL_CROP','TREAT_LAST_QKV_LORA4')
COLS = ['sample_index','row_id','family','role','available','target','logit0','logit1','selected_family_logit','prediction']
SAMPLE_KEYS = {'full_view','central_view','fractions','central4','condition','target','family','fit','available','row_id'}


def need(ok, message):
    if not bool(ok): raise ValueError(message)


def safe(path):
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).casefold() and '__pycache__' not in p.parts,'Forbidden path before filesystem')
    for q in list(p.parents)[::-1]+[p]:
        try: mode = os.lstat(q).st_mode
        except FileNotFoundError: continue
        need(not stat.S_ISLNK(mode),'Symlink rejected')
    return p


def raw(path):
    with os.fdopen(os.open(safe(path),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f: return f.read()


def binding(path):
    b=raw(path)
    return dict(path=str(path),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))


def same(a,b):
    return a['path']==b['path'] and a['sha256']==b['sha256']


def checked(b):
    v=raw(b['path'])
    need(hashlib.sha256(v).hexdigest()==b['sha256'],'Pinned SHA changed')
    if 'size_bytes' in b: need(len(v)==b['size_bytes'],'Pinned size changed')
    return v


def json_write(path,value):
    with safe(path).open('x',encoding='utf-8') as f: f.write(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')


def bool_csv(v):
    need(v in ('True','False'),'Canonical CSV boolean')
    return v=='True'


def metrics(rows, arm):
    cm=[[sum(r['target']==y and r[arm+'_prediction']==p for r in rows) for p in ('D','E','U')] for y in (0,1)]
    result=dict(N=len(rows),CM_D_E_by_D_E_U=cm,U=sum(r[2] for r in cm),U_rate=sum(r[2] for r in cm)/len(rows) if rows else None)
    for name,k in (('D',0),('E',1)):
        truth=sum(cm[k]);predicted=cm[0][k]+cm[1][k];tp=cm[k][k]
        result[name]=dict(truth=truth,predicted=predicted,precision=tp/predicted if predicted else None,recall=tp/truth if truth else None)
    recalls=[result[k]['recall'] for k in ('D','E')]
    result['balanced_recall']=sum(recalls)/2 if all(v is not None for v in recalls) else None
    return result


def paired(rows):
    out={}
    for label,subset in [('ALL',rows)]+[(name,[r for r in rows if r['target']==y]) for y,name in enumerate(('D','E'))]:
        counts=Counter()
        for r in subset:
            truth=('D','E')[r['target']];a=r['control_prediction']==truth;b=r['treatment_prediction']==truth
            counts['both_correct' if a and b else 'damaged' if a else 'recovered' if b else 'both_wrong']+=1
            counts['prediction_changed']+=r['control_prediction']!=r['treatment_prediction']
        out[label]=dict(N=len(subset),**{k:counts[k] for k in ('both_correct','both_wrong','recovered','damaged','prediction_changed')},net_correct=counts['recovered']-counts['damaged'])
    return out


def reduce_saved(rows):
    by_family={};syn_distance={};dev_pairs={}
    for family,name in ((0,'SYN_MAIN'),(1,'YCB_AUX')):
        by_family[name]={}
        for role in ('FIT','DEV'):
            rr=[r for r in rows if r['family']==family and r['role']==role]
            by_family[name][role]={arm:metrics(rr,arm) for arm in ('control','treatment')}
        dev_pairs[name]=paired([r for r in rows if r['family']==family and r['role']=='DEV'])
    for role in ('FIT','DEV'):
        syn_distance[role]={}
        for label in ('LE3','GT3','UNKNOWN'):
            rr=[r for r in rows if r['family']==0 and r['role']==role and (('UNKNOWN' if r['q_reference_distance_px'] is None else 'LE3' if r['q_reference_distance_px']<=3 else 'GT3')==label)]
            syn_distance[role][label]={arm:metrics(rr,arm) for arm in ('control','treatment')}
    recalls={a:[by_family[f]['DEV'][a][c]['recall'] for f in ('SYN_MAIN','YCB_AUX') for c in ('D','E')] for a in ('control','treatment')}
    defined=all(v is not None for values in recalls.values() for v in values)
    delta=[b-a for a,b in zip(recalls['control'],recalls['treatment'])] if defined else None
    gain=sum(delta)/4 if defined else None
    gate=dict(defined=defined,macro_DEV_balanced_recall_gain=gain,macro_gain_pp=None if gain is None else gain*100,
              class_order=['SYN_D','SYN_E','YCB_D','YCB_E'],class_recall_deltas=delta,
              required_macro_gain=.03,max_class_recall_decrease=.02,
              numeric_gate_pass=bool(defined and gain>=.03-1e-12 and min(delta)>=-.02-1e-12),
              ancestry='UNKNOWN',current_admission=False,selected_arm=None,model_adoption=False)
    return dict(family_metrics=by_family,SYN_q_reference_distance=syn_distance,DEV_paired=dev_pairs,gate=gate)


def parse_arm(b, samples):
    reader=csv.DictReader(io.StringIO(checked(b).decode('utf-8')))
    need(reader.fieldnames==COLS,'Exact producer CSV schema')
    rows=list(reader);need(len(rows)==len(samples),'All7617 predictions retained')
    for i,(r,s) in enumerate(zip(rows,samples)):
        need(int(r['sample_index'])==i and r['row_id']==s['row_id'] and int(r['family'])==s['family'] and r['role']==s['role'] and int(r['target'])==s['target'] and bool_csv(r['available'])==s['available'],'CSV row/target/family/role/availability identity')
        if s['available']:
            values=[float(r[k]) for k in ('logit0','logit1','selected_family_logit')]
            need(all(math.isfinite(v) for v in values) and values[2]==values[s['family']],'Finite selected-family logit')
            need(r['prediction']==('E' if values[2]>=0 else 'D'),'Fixed threshold zero')
        else:
            need(r['prediction']=='U' and all(r[k]=='' for k in ('logit0','logit1','selected_family_logit')),'Unavailable U with missing logits; never dropped')
    return rows


def report(summary):
    def pct(x): return '—' if x is None else f'{100*x:.2f}%'
    lines=['# V233: 같은 local crop에서 고정 backbone과 마지막 qkv 적응 대조','','두 referent를 분리한 SOURCE 결과다. 현재 실사 성공·모델 채택 결과가 아니다. 원 query/L/native와 문턱0을 유지했고 추가 평가 기반 조정은 하지 않았다.','','원15040행은 SYN7040+YCB8000이다. known D/E7617행은 사용 불가 입력도 삭제하지 않고 양팔 출력에 보존했다. 나머지7423행은 원 S/U/제외 타깃−1이며 forward 대상이 아니다. 이7423은 모델 U가 아니다.','','SYN main은 원영상 native witness0의 associated-point D/E다. literal q-center owner가 아니다. YCB AUX는 associated-r surface D/E이며 native UNKNOWN이다. 두 CM을 하나의 의미 과업으로 합치지 않는다.','','|가족|역할|팔|N|CM [[DD,DE,DU],[ED,EE,EU]]|D P/R|E P/R|U율|균형 recall|','|---|---|---|---:|---|---|---|---|---|']
    for family,roles in summary['family_metrics'].items():
        for role,arms in roles.items():
            for arm,m in arms.items(): lines.append(f"|{family}|{role}|{arm}|{m['N']}|{m['CM_D_E_by_D_E_U']}|{pct(m['D']['precision'])}/{pct(m['D']['recall'])}|{pct(m['E']['precision'])}/{pct(m['E']['recall'])}|{pct(m['U_rate'])}|{pct(m['balanced_recall'])}|")
    lines+=['','|DEV 가족|범위|회복|악화|순증|양팔 모두정답|양팔 모두오답|','|---|---|---:|---:|---:|---:|---:|']
    for family,groups in summary['DEV_paired'].items():
        for label,c in groups.items(): lines.append(f"|{family}|{label}|{c['recovered']}|{c['damaged']}|{c['net_correct']}|{c['both_correct']}|{c['both_wrong']}|")
    g=summary['gate'];gain='정의불가' if g['macro_gain_pp'] is None else f"{g['macro_gain_pp']:.3f}pp"
    lines+=['',f"사전 수치 gate는 DEV macro balanced recall +3pp 이상이며 각 family/class recall 감소가2pp 이하여야 한다. 관측 gain={gain}, 수치 통과={g['numeric_gate_pass']}. 계산 결과와 무관하게 SYN donor/background와 YCB DEV의 교차 계보 독립성은 UNKNOWN이므로 current admission=false다. 새 팔·seed·epoch·문턱·데이터 선택은 하지 않았다.",'','SYN q–reference ≤3px/>3px 표는 target reference와 원 query의 거리에 따른 서술이며 좌표오차 보정이나 q-center ownership 증거가 아니다. YCB target sidecar에는 r 좌표가 없어 거리를 null로 보존한다. SOURCE DEV는 사전 노출/재사용됐고 fresh blind 평가가 아니다.','',summary['execution_provenance_note'],'','[전체 검산 수치](SUMMARY.json) · [strata CM/P/R/U](METRICS.csv) · [7617행 paired 결과](PAIRED_ALL7617.csv) · [원본 binding/검산 범위](COMPLETE.json)','','같은 배포 입력에서 실제 새 모델 성능이 개선됐다는 결론은 이 SOURCE 결과로 내리지 않는다.','']
    return '\n'.join(lines)


def metric_table(summary):
    table=[]
    def add(family,role,stratum,arms):
        for arm,m in arms.items():
            c=m['CM_D_E_by_D_E_U'];table.append(dict(family=family,role=role,stratum=stratum,arm=arm,N=m['N'],DD=c[0][0],DE=c[0][1],DU=c[0][2],ED=c[1][0],EE=c[1][1],EU=c[1][2],D_precision=m['D']['precision'],D_recall=m['D']['recall'],E_precision=m['E']['precision'],E_recall=m['E']['recall'],U_rate=m['U_rate'],balanced_recall=m['balanced_recall']))
    for family,roles in summary['family_metrics'].items():
        for role,arms in roles.items():add(family,role,'ALL',arms)
    for role,strata in summary['SYN_q_reference_distance'].items():
        for name,arms in strata.items():add('SYN_MAIN',role,name,arms)
    return table


def run(entry_path):
    import numpy as np  # NPZ는 실제 Root 완료 승인 후 이 경로에서만 읽는다.
    e=json.loads(raw(entry_path));need(e.get('approved_completed_SOURCE_saved_summary') is True,'Root completed-fit summary entry required')
    need(same(e['code'],binding(Path(__file__).absolute())) and same(e['root_protocol'],PROTOCOL),'Code/root protocol pin')
    protocol=json.loads(checked(PROTOCOL));need(protocol['source_gate_before_results']['treatment_gate'].startswith('macro balanced-recall gain>=3pp'),'Frozen gate protocol')
    fit_complete=Path(e['inputs']['fit_complete']['path']);fit=fit_complete.parent
    need(fit.is_relative_to(R/'V233_ROOT_CROP_ADAPTATION') and fit_complete.name=='COMPLETE.json','Explicit root V233 saved fit directory')
    fixed={'fit_complete':fit_complete,'all2_freeze':fit/'ALL2_FINAL1000_FROZEN_BEFORE_DEV_FORWARD.json','crop_complete':CROP/'COMPLETE.json','samples':CROP/'samples.npz','sample_diagnostics':CROP/'SAMPLE_DIAGNOSTICS.jsonl','full_rows':CROP/'ALL15040_ROWS.jsonl'}
    need(set(e['inputs'])==set(fixed),'Closed six saved-input bindings')
    for k,p in fixed.items():need(e['inputs'][k]['path']==str(p),'Exact saved path '+k)
    f=json.loads(checked(e['inputs']['fit_complete']));need(f.get('completed') is True and f.get('PASS') is True and f['steps_each']==1000 and f['DEV_forward_before_freeze']==0,'Completed1000 both-final gate')
    need(same(f['entry'],e['training_entry']) and same(f['code'],e['training_code']) and same(f['samples_npz'],e['inputs']['samples']) and same(f['all2_freeze'],e['inputs']['all2_freeze']),'Frozen fit input/binding closure')
    need(set(e['arm_outputs'])==set(ARMS) and set(f['outputs'])==set(ARMS),'Both fixed arms retained')
    for a in ARMS:need(same(e['arm_outputs'][a],f['outputs'][a]) and e['arm_outputs'][a]['path']==str(fit/(a+'_ALL_SOURCE_ROWS.csv')),'Exact own CSV binding')
    freeze=json.loads(checked(e['inputs']['all2_freeze']))
    need(freeze.get('completed') is True and freeze.get('PASS') is True and freeze['steps_each']==1000 and freeze['DEV_forward_calls_before_freeze']==0,'Both-final freeze before DEV')
    need(freeze['checkpoints']==f['checkpoints'] and set(freeze['checkpoints'])==set(ARMS),'Checkpoint metadata closure; no model bytes read')
    for a,b in [('same_head_initial_sha256','head_initial_sha256'),('same_draws_sha256','draws_sha256'),('same_base_sha256','base_state_sha256_before_after')]:need(freeze[a]==f[b],'Shared frozen identity '+a)
    crop=json.loads(checked(e['inputs']['crop_complete']));need(crop.get('completed') is True and crop.get('PASS') is True and crop['samples']==7617 and crop['all_rows']==15040,'Completed source data closure')
    outputs={b['path']:b for b in crop['outputs']}
    for k in ('samples','sample_diagnostics','full_rows'):need(e['inputs'][k]['path'] in outputs and same(e['inputs'][k],outputs[e['inputs'][k]['path']]),'Data output exact binding '+k)
    with np.load(io.BytesIO(checked(e['inputs']['samples'])),allow_pickle=False) as z:
        need(set(z.files)==SAMPLE_KEYS,'Exact sample metadata NPZ only')
        data={k:z[k] for k in z.files}
    n=len(data['row_id']);need(n==7617 and data['target'].dtype==np.int8 and data['family'].dtype==np.int8 and data['available'].dtype==data['fit'].dtype==bool,'Sample census/types')
    diag=[json.loads(l) for l in checked(e['inputs']['sample_diagnostics']).splitlines()]
    full=[json.loads(l) for l in checked(e['inputs']['full_rows']).splitlines()]
    need(len(diag)==n and len(full)==15040 and len({r['row_id'] for r in full})==15040,'All source rows preserved')
    need(Counter((r['family_code'],r['target']) for r in full)=={(0,0):5853,(0,1):243,(0,-1):944,(1,0):1181,(1,1):340,(1,-1):6479},'Both referents original target census')
    all_by_id={r['row_id']:r for r in full};samples=[]
    for i,d in enumerate(diag):
        need(d==all_by_id[d['row_id']] and d['sample_index']==i and str(data['row_id'][i])==d['row_id'],'Sidecar/fullledger/sample order identity')
        family=int(data['family'][i]);target=int(data['target'][i]);fit=bool(data['fit'][i]);av=bool(data['available'][i])
        need(family==d['family_code'] and target==d['target'] and target in (0,1) and fit==d['fit'] and av==d['available'],'Sample target/family/role/availability exact')
        need(int(data['full_view'][i])==2*i and int(data['central_view'][i])==2*i+1 and bool(data['central4'][i])==d['central4'] and np.array_equal(data['fractions'][i],np.asarray(d['valid_fractions'],np.float32)),'Original input metadata retained')
        distance=d['q_reference_distance_px_diagnostic_only']
        if family==1:need(distance is None and d['reference_xy_GT_diagnostic_only'] is None,'YCB r absent; never infer')
        elif distance is not None:need(math.isfinite(distance) and distance>=0 and math.isclose(distance,math.dist(d['query_xy'],d['reference_xy_GT_diagnostic_only']),rel_tol=1e-12,abs_tol=1e-12),'SYN q/reference sidecar algebra')
        samples.append(dict(sample_index=i,row_id=d['row_id'],family=family,target=target,role='FIT' if fit else 'DEV',available=av,q_reference_distance_px=distance))
    need(all(full[r]['target']==-1 for r in range(len(full)) if full[r]['sample_index'] is None) and sum(r['sample_index'] is None for r in full)==7423,'7423 masked not forwarded; distinct from prediction U')
    need(all(data['fit'][i] and data['available'][i] and data['target'][i]>=0 for i in freeze['prebarrier_only_known_available_FIT_samples']),'Prebarrier sample metadata remains FIT only')
    arms={a:parse_arm(e['arm_outputs'][a],samples) for a in ARMS};rows=[]
    for i,s in enumerate(samples):
        r=dict(s)
        for a,short in zip(ARMS,('control','treatment')):
            ar=arms[a][i];r[short+'_prediction']=ar['prediction'];r[short+'_logit']=None if not ar['selected_family_logit'] else float(ar['selected_family_logit'])
        r['logit_delta']=None if r['control_logit'] is None else r['treatment_logit']-r['control_logit'];rows.append(r)
    summary=dict(completed=True,PASS=True,all_source_rows=15040,known_primary=7617,masked_not_forward=7423,threshold_logit0=True,**reduce_saved(rows),two_referents_not_merged=True,model_or_threshold_selected=False,new_current_inference=0,
                 execution_provenance_note=e['execution_provenance_note'])
    out=safe(e['output_dir']);need(out==H/'results' and not out.exists(),'Fresh fixed summary output');out.mkdir()
    with (out/'PAIRED_ALL7617.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    table=metric_table(summary)
    with (out/'METRICS.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    json_write(out/'SUMMARY.json',summary)
    with (out/'PORTABLE_REPORT_KO.md').open('x',encoding='utf-8') as f:f.write(report(summary))
    complete=dict(completed=True,PASS=True,code=binding(Path(__file__).absolute()),root_protocol=PROTOCOL,entry=binding(entry_path),inputs=e['inputs'],arm_outputs=e['arm_outputs'],training_code=e['training_code'],training_entry=e['training_entry'],runtime_repair_provenance=e.get('runtime_repair_provenance'),
                  outputs=[binding(out/n) for n in ('SUMMARY.json','METRICS.csv','PAIRED_ALL7617.csv','PORTABLE_REPORT_KO.md')],scope='Saved CSV/NPZ metadata/JSONL arithmetic and frozen metadata closure; no checkpoint decoding or independent forward/loss replay',model_prefix_RGB_raw_GT_reads=0,current_admission=False)
    json_write(out/'COMPLETE.json',complete)
    print(json.dumps(dict(completed=True,PASS=True,complete=binding(out/'COMPLETE.json'))))


def verify_saved():
    # 공개 package에서 stdlib만으로 출력 표를 재집계한다. absolute binding은 따라가지 않는다.
    d=Path(__file__).absolute().parent/'results';s=json.loads(raw(d/'SUMMARY.json'))
    rows=list(csv.DictReader(io.StringIO(raw(d/'PAIRED_ALL7617.csv').decode())))
    for r in rows:
        for k in ('family','target'):r[k]=int(r[k])
        need(r['family'] in (0,1) and r['target'] in (0,1) and r['role'] in ('FIT','DEV'),'Saved family/role/known target')
        r['available']=bool_csv(r['available']);r['q_reference_distance_px']=None if r['q_reference_distance_px']=='' else float(r['q_reference_distance_px'])
        values=[]
        for a in ('control','treatment'):
            p=r[a+'_prediction'];v=None if r[a+'_logit']=='' else float(r[a+'_logit'])
            need(v is None or math.isfinite(v),'Saved finite logit')
            need(p==('U' if v is None else 'E' if v>=0 else 'D') and (p=='U')==(not r['available']),'Saved zero threshold/availability')
            values.append(v)
        need(r['logit_delta']=='' if values[0] is None else float(r['logit_delta'])==values[1]-values[0],'Saved paired logit delta')
    need([int(r['sample_index']) for r in rows]==list(range(7617)),'Saved sample order')
    need(len(rows)==7617 and len({r['row_id'] for r in rows})==7617,'Saved known census')
    actual=reduce_saved(rows)
    need(all(s[k]==v for k,v in actual.items()),'Saved summary arithmetic replay')
    table=list(csv.DictReader(io.StringIO(raw(d/'METRICS.csv').decode())))
    expected=[{k:'' if v is None else str(v) for k,v in row.items()} for row in metric_table(actual)]
    need(table==expected,'Saved all CM/P/R/U table arithmetic')
    print(json.dumps(dict(completed=True,PASS=True,rows=7617,scope='stdlib paired CSV arithmetic only; original input/GT/models not replayed')))


def self_test():
    rows=[dict(row_id=str(i),family=f,target=y,role='DEV',available=True,q_reference_distance_px=3 if f==0 else None,control_prediction=p,treatment_prediction=t) for i,(f,y,p,t) in enumerate([(0,0,'E','D'),(0,1,'E','E'),(1,0,'D','E'),(1,1,'U','E')])]
    result=reduce_saved(rows)
    need(metrics(rows[:2],'control')['CM_D_E_by_D_E_U']==[[0,1,0],[0,1,0]],'Toy 2x3')
    need(result['DEV_paired']['SYN_MAIN']['ALL']['recovered']==1 and result['DEV_paired']['YCB_AUX']['ALL']['damaged']==1,'Toy paired recovery/damage')
    need(not result['gate']['numeric_gate_pass'] and result['gate']['current_admission'] is False,'Toy class harm blocks numeric gate; ancestry blocks admission')
    need(result['SYN_q_reference_distance']['DEV']['LE3']['control']['N']==2,'Closed3px boundary')
    print(json.dumps(dict(completed=True,PASS=True,checks=4,actual_results_models_or_payloads_read=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',type=Path);p.add_argument('--self-test',action='store_true');p.add_argument('--verify-saved',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    elif a.verify_saved:verify_saved()
    else:need(a.entry is not None,'--entry required');run(a.entry)
