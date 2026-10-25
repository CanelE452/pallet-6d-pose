"""V238 원 SOURCE7617 저장 출력 산술. 모델/픽셀/특징/GT 실행 없음."""
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

R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=R/'V238_SOURCE_REPORT_C'
CROP=R/'V233_ROOT_CROP_ADAPTATION/crop_samples_v1'
PROTOCOL=dict(path=str(R/'V238_ROOT_CONTEXT_RESIDUAL/PROTOCOL_BEFORE_NEW_FEATURES_OR_FIT.json'),sha256='17b3b00b1c8b801dc536460f555d40ff8df37f6fe2355955a5e1ad66ecb29ba3')
BASELINE_SUMMARY=dict(path=str(R/'V233_SOURCE_REPORT_C/results/SUMMARY.json'),sha256='8a9b54c59ef94639815e5bd575a48433fc83d52e3776140d16c86c5782546dc4')
ARMS=('CONTROL_LOCAL_CROP','TREAT_CONTEXT_RESIDUAL')
COLS=['sample_index','row_id','family','role','available','target','logit0','logit1','selected_family_logit','prediction']

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

def entry_authority(e):
    need(e.get('approved_completed_SOURCE_saved_summary') is True,'Root completed-fit report entry required; no actual CSV access before it')

def reduce_v238(rows):
    result=reduce_saved(rows)
    g=result['gate'];old_numeric=g['numeric_gate_pass'];g.pop('ancestry',None)
    recall=result['family_metrics']['YCB_AUX']['DEV']['treatment']['E']['recall']
    g.update(mean_gain_and_class_drop_gate_pass=old_numeric,prospective_YCB_DEV_E_recall_floor=.5,YCB_DEV_E_recall=recall,
             YCB_DEV_E_recall_floor_pass=bool(recall is not None and recall>=.5),
             numeric_gate_pass=bool(old_numeric and recall is not None and recall>=.5),current_admission=False,
             selected_arm=None,model_adoption=False)
    return result


def report(summary):
    pct=lambda x:'—' if x is None else f'{100*x:.2f}%'
    lines=['# V238: 원 SOURCE에서 q8 관계 context residual 대조','',
           'SYN main은 intended-native witness0 associated-point D/E, YCB AUX는 associated-r surface D/E(native UNKNOWN)다. 두 referent의 결과를 따로 표시한다. 원 query/L/native·availability·FIT/DEV·문턱0을 유지했다. SOURCE는 이미 노출된 진단이며 fresh-blind 평가나 새 current 실사 개선이 아니다.','',
           '원15040행 중 known D/E7617행을 원 sample_index 순서로 보존했다. 나머지7423행은 원 S/U/제외 target−1로서 모델 U가 아니다. 새 V235 crop-paste 증강은0이며 원 natural/SYN actual scene만 평가한다.','',
           '|가족|역할|팔|N|CM [[DD,DE,DU],[ED,EE,EU]]|D P/R|E P/R|U율|균형 recall|','|---|---|---|---:|---|---|---|---|---|']
    for family,roles in summary['family_metrics'].items():
        for role,arms in roles.items():
            for arm,m in arms.items():lines.append(f"|{family}|{role}|{arm}|{m['N']}|{m['CM_D_E_by_D_E_U']}|{pct(m['D']['precision'])}/{pct(m['D']['recall'])}|{pct(m['E']['precision'])}/{pct(m['E']['recall'])}|{pct(m['U_rate'])}|{pct(m['balanced_recall'])}|")
    lines+=['','|DEV 가족|정답 범위|회복|악화|순증|모두 정답|모두 오답|','|---|---|---:|---:|---:|---:|---:|']
    for family,groups in summary['DEV_paired'].items():
        for label,c in groups.items():lines.append(f"|{family}|{label}|{c['recovered']}|{c['damaged']}|{c['net_correct']}|{c['both_correct']}|{c['both_wrong']}|")
    g=summary['gate'];gain='정의불가' if g['macro_gain_pp'] is None else f"{g['macro_gain_pp']:.3f}pp"
    lines+=['',f"사전 SOURCE gate: 두 family DEV balanced recall 평균 +3pp 이상, 각 family/class recall 감소2pp 이하, YCB DEV E recall50% 이상. 관측 gain={gain}, 평균/감소 조건={g['mean_gain_and_class_drop_gate_pass']}, YCB E recall={pct(g['YCB_DEV_E_recall'])}, E floor={g['YCB_DEV_E_recall_floor_pass']}, 전체 수치 gate={g['numeric_gate_pass']}. 이 계산으로 모델·seed·문턱을 선택하지 않으며 current admission=false, selected arm=null을 유지한다.",'',
           'CONTROL은 원 V233 final1000 bytes를 재사용하여 새 optimizer update0회다. TREAT는 원 base initialseed17+zero-final residual에서1000회이며 원PCG17 draw ledger를 쓴다. 기존base76098에residual9956을추가한total86054다. encoder/LoRA 새학습parameter는0이다. 용량이달라parameter-matched 인과시험은아니다. 두final checkpoint와newhead DEV-before-freeze0을확인한후저장CSV를읽었다. V233 runtime수리나V235증강실험을이번trial의실패/성공으로옮기지않는다.','',
           'FIT/DEV active descriptor QA 및CONTROL replay는봉인producer metadata의정확binding으로인용한다. 본검산은신경망forward/특징/weights를다시읽지않았다. attention other7의값이별도기록됐더라도이보고서의primary CM과새teacher정답에합치지않으며여기서독립재연산하지않았다.','',
           'SYN q-reference≤3px/>3px 표는원sidecar의거리진단이며query보정이나literal-q owner gold가아니다. 원YCB sidecar의reference/거리는null로유지했다. 입력q자체의YCB GT-jitter emulation 출처는감추지않지만GTref/pose/ownership을추가forward입력으로허용한것은아니다.','',
           '[수치 JSON](SUMMARY.json) · [CM/P/R/U·SYN거리 strata](METRICS.csv) · [원7617 paired 출력](PAIRED_ALL7617.csv) · [binding·검산 범위](COMPLETE.json)','',
           '새 current/held 추론·PnP·채택은없다. SOURCE에서보이는변화만으로실사문제해결이나attention의인과효과를주장하지않는다.','']
    return '\n'.join(lines)


def run(entry_path):
    e=json.loads(raw(entry_path));entry_authority(e)
    need(same(e['code'],binding(Path(__file__).absolute())) and same(e['root_protocol'],PROTOCOL),'Pinned report CODE/root protocol')
    protocol=json.loads(checked(PROTOCOL));gate=protocol['fixed_SOURCE_gate']
    need(gate['mean_family_DEV_balanced_recall_gain_min_pp']==3 and gate['max_family_class_D_E_recall_drop_pp']==2 and gate['prospective_YCB_DEV_E_recall_floor_percent']==50,'Fixed three-part SOURCE gate')
    need(protocol['model']['threshold_logit']==0 and protocol['model']['base_parameters']==76098 and protocol['model']['new_residual_parameters']==9956 and protocol['model']['total_parameters']==86054,'Frozen model/threshold metadata contract')
    need(set(e['inputs'])=={'fit_complete','freeze','crop_complete','sample_diagnostics','full_rows','baseline_summary'},'Closed saved input metadata set')
    f=json.loads(checked(e['inputs']['fit_complete']))
    need(f.get('completed') is True and f.get('PASS') is True and f['new_CONTROL_optimizer_updates']==0 and f['new_TREAT_optimizer_updates']==1000 and f['new_head_DEV_forward_before_freeze']==0,'Completed final1000; CONTROL reuse')
    need(f['head_parameters']==dict(zip(ARMS,(76098,86054))) and f['new_DINO_or_LoRA_parameters']==0 and f['new_V235_CF']==0 and f['current_admission'] is False,'Residual capacity/natural source-only contract')
    need(same(f['code'],e['training_code']) and same(f['entry'],e['training_entry']) and same(f['freeze'],e['inputs']['freeze']),'Pinned new training completion closure')
    need(set(f['outputs'])==set(ARMS) and set(f['checkpoints'])==set(ARMS) and set(e['arm_outputs'])==set(ARMS),'Both fixed arms retained')
    fit=Path(e['inputs']['fit_complete']['path']).parent
    need(fit.is_relative_to(R) and any(p.startswith('V238_') for p in fit.parts),'Explicit V238 result folder')
    need(e['inputs']['freeze']['path']==str(fit/'FINAL1000_FROZEN_BEFORE_NEW_HEAD_DEV_FORWARD.json'),'Exact freeze basename')
    for arm in ARMS:need(same(e['arm_outputs'][arm],f['outputs'][arm]) and e['arm_outputs'][arm]['path']==str(fit/(arm+'_ALL_SOURCE_ROWS.csv')),'Exact own CSV binding '+arm)
    freeze=json.loads(checked(e['inputs']['freeze']))
    need(freeze.get('completed') is True and freeze.get('PASS') is True and freeze['new_CONTROL_optimizer_updates']==0 and freeze['new_TREAT_optimizer_updates']==1000 and freeze['new_head_DEV_forward_before_freeze']==0 and freeze['new_V235_CF']==0,'Both-final/newhead-beforeDEV/natural gate')
    need(freeze['original_prefix_numeric_rows_before_freeze_only_FIT'] is True and freeze['frozen_NODE_context_features_may_include_DEV'] is True,'Prefix FIT policy; frozen features not new head output')
    need(freeze['checkpoints']==f['checkpoints'] and freeze['same_original_draws_sha256']==f['original_draws_sha256'] and freeze['original_initial_base_sha256']==f['original_initial_base_sha256'],'Frozen checkpoint/draw/base metadata closure')
    for key in ('CONTROL_reuse_QA','CONTROL_replay_QA','FIT_active_descriptor_QA','DEV_active_descriptor_QA'):need(same(f[key],e[key]),'QA metadata binding '+key)
    need(same(freeze['CONTROL_reuse_QA'],f['CONTROL_reuse_QA']) and same(freeze['FIT_active_descriptor_QA'],f['FIT_active_descriptor_QA']),'Prebarrier QA bindings')
    crop=json.loads(checked(e['inputs']['crop_complete']))
    need(e['inputs']['crop_complete']['path']==str(CROP/'COMPLETE.json') and e['inputs']['crop_complete']['sha256']==protocol['data']['source_crop_complete_sha256'],'Unchanged original crop binding')
    need(crop.get('completed') is True and crop.get('PASS') is True and crop['samples']==7617 and crop['all_rows']==15040,'Original source census')
    cb={x['path']:x for x in crop['outputs']}
    for key,name in [('sample_diagnostics','SAMPLE_DIAGNOSTICS.jsonl'),('full_rows','ALL15040_ROWS.jsonl')]:need(e['inputs'][key]['path']==str(CROP/name) and same(e['inputs'][key],cb[str(CROP/name)]),'Original sidecar pin '+key)
    diag=[json.loads(x) for x in checked(e['inputs']['sample_diagnostics']).splitlines()]
    full=[json.loads(x) for x in checked(e['inputs']['full_rows']).splitlines()]
    need(len(diag)==7617 and len(full)==15040 and len({x['row_id'] for x in full})==15040,'All original SOURCE rows')
    need(Counter((x['family_code'],x['target']) for x in full)=={(0,0):5853,(0,1):243,(0,-1):944,(1,0):1181,(1,1):340,(1,-1):6479},'Original labels not modified')
    full_by_id={x['row_id']:x for x in full};samples=[]
    for i,d in enumerate(diag):
        need(d==full_by_id[d['row_id']] and d['sample_index']==i and d['target'] in (0,1) and d['family_code'] in (0,1),'Original sidecar row/target identity')
        dist=d['q_reference_distance_px_diagnostic_only']
        if d['family_code']==1:need(dist is None and d['reference_xy_GT_diagnostic_only'] is None,'Original YCB null kept')
        elif dist is not None:need(math.isfinite(dist) and dist>=0 and math.isclose(dist,math.dist(d['query_xy'],d['reference_xy_GT_diagnostic_only']),rel_tol=1e-12,abs_tol=1e-12),'Original SYN diagnostic distance')
        samples.append(dict(sample_index=i,row_id=d['row_id'],family=d['family_code'],target=d['target'],role='FIT' if d['fit'] else 'DEV',available=d['available'],q_reference_distance_px=dist))
    need(Counter((x['family'],x['role']) for x in samples)=={(0,'FIT'):4752,(0,'DEV'):1344,(1,'FIT'):1124,(1,'DEV'):397},'Original family/role denominators')
    need(sum(x['sample_index'] is None for x in full)==7423 and all(x['target']==-1 for x in full if x['sample_index'] is None),'Masked retained; not model U')
    arms={arm:parse_arm(e['arm_outputs'][arm],samples) for arm in ARMS};rows=[]
    for i,s in enumerate(samples):
        r=dict(s)
        for arm,short in zip(ARMS,('control','treatment')):
            ar=arms[arm][i];r[short+'_prediction']=ar['prediction'];r[short+'_logit']=None if ar['selected_family_logit']=='' else float(ar['selected_family_logit'])
        r['logit_delta']=None if r['control_logit'] is None else r['treatment_logit']-r['control_logit'];rows.append(r)
    reduction=reduce_v238(rows)
    need(same(e['inputs']['baseline_summary'],BASELINE_SUMMARY),'Pinned original V233 CONTROL summary; no old treatment merge')
    baseline=json.loads(checked(BASELINE_SUMMARY))
    for family in ('SYN_MAIN','YCB_AUX'):
        for role in ('FIT','DEV'):need(reduction['family_metrics'][family][role]['control']==baseline['family_metrics'][family][role]['control'],'Original CONTROL scalar metrics exact')
    summary=dict(completed=True,PASS=True,all_source_rows=15040,known_primary=7617,masked_not_forward=7423,new_V235_CF=0,new_CONTROL_optimizer_updates=0,new_TREAT_optimizer_updates=1000,head_parameters=f['head_parameters'],base_parameters=76098,residual_parameters=9956,total_parameters=86054,new_DINO_or_LoRA_parameters=0,threshold_logit0=True,two_referents_not_merged=True,model_or_threshold_selected=False,new_current_inference=0,attention_other7_independently_recomputed=False,**reduction)
    out=safe(e['output_dir']);need(out==H/'results' and not out.exists(),'Fresh report output');out.mkdir()
    with (out/'PAIRED_ALL7617.csv').open('x',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    table=metric_table(summary)
    with (out/'METRICS.csv').open('x',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    json_write(out/'SUMMARY.json',summary)
    with (out/'PORTABLE_REPORT_KO.md').open('x',encoding='utf-8') as file:file.write(report(summary))
    complete=dict(completed=True,PASS=True,code=binding(Path(__file__).absolute()),root_protocol=PROTOCOL,entry=binding(entry_path),fit_complete=e['inputs']['fit_complete'],inputs=e['inputs'],arm_outputs=e['arm_outputs'],training_code=e['training_code'],training_entry=e['training_entry'],CONTROL_reuse_QA=e['CONTROL_reuse_QA'],CONTROL_replay_QA=e['CONTROL_replay_QA'],FIT_active_descriptor_QA=e['FIT_active_descriptor_QA'],DEV_active_descriptor_QA=e['DEV_active_descriptor_QA'],outputs=[binding(out/name) for name in ('SUMMARY.json','METRICS.csv','PAIRED_ALL7617.csv','PORTABLE_REPORT_KO.md')],scope='Saved original SOURCE CSV/JSONL/metadata arithmetic only; no model/feature/NPZ/RGB/GT decode or forward.',current_admission=False)
    json_write(out/'COMPLETE.json',complete)
    print(json.dumps(dict(completed=True,PASS=True,complete=binding(out/'COMPLETE.json'))))


def verify_saved():
    # 재현은 package 상대 results만 읽는 stdlib 산술이다. 절대 binding은 따라가지 않는다.
    d=Path(__file__).absolute().parent/'results';summary=json.loads(raw(d/'SUMMARY.json'))
    rows=list(csv.DictReader(io.StringIO(raw(d/'PAIRED_ALL7617.csv').decode())))
    for r in rows:
        for k in ('sample_index','family','target'):r[k]=int(r[k])
        need(r['family'] in (0,1) and r['target'] in (0,1) and r['role'] in ('FIT','DEV'),'Saved known identity')
        r['available']=bool_csv(r['available']);r['q_reference_distance_px']=None if r['q_reference_distance_px']=='' else float(r['q_reference_distance_px'])
        need(r['family']==0 or r['q_reference_distance_px'] is None,'Original YCB null')
        logits=[]
        for arm in ('control','treatment'):
            z=None if r[arm+'_logit']=='' else float(r[arm+'_logit']);pred=r[arm+'_prediction']
            need(z is None or math.isfinite(z),'Saved finite logits')
            need(pred==('U' if z is None else 'E' if z>=0 else 'D') and (pred=='U')==(not r['available']),'Saved original U/threshold0')
            logits.append(z)
        need(r['logit_delta']=='' if logits[0] is None else float(r['logit_delta'])==logits[1]-logits[0],'Saved logit delta')
    need(len(rows)==7617 and len({r['row_id'] for r in rows})==7617 and [r['sample_index'] for r in rows]==list(range(7617)),'Original known7617 order')
    actual=reduce_v238(rows);need(all(summary[k]==v for k,v in actual.items()),'Saved metrics/paired/three-part gate arithmetic')
    table=list(csv.DictReader(io.StringIO(raw(d/'METRICS.csv').decode())))
    expected=[{k:'' if v is None else str(v) for k,v in row.items()} for row in metric_table(actual)]
    need(table==expected,'Saved all strata metrics arithmetic')
    print(json.dumps(dict(completed=True,PASS=True,rows=7617,scope='stdlib saved output arithmetic only; no model/input binding follow')))


def self_test():
    rows=[]
    for family in (0,1):
        for target in (0,1):
            for k in (0,1):
                control=('D','E')[target] if k==0 and not(family==1 and target==1) else ('E','D')[target]
                treat=('D','E')[target] if not(family==1 and target==1) else 'D'
                rows.append(dict(row_id=str(len(rows)),family=family,target=target,role='DEV',available=True,q_reference_distance_px=3 if family==0 else None,control_prediction=control,treatment_prediction=treat))
    r=reduce_v238(rows)
    need(r['gate']['mean_gain_and_class_drop_gate_pass'] and not r['gate']['YCB_DEV_E_recall_floor_pass'] and not r['gate']['numeric_gate_pass'],'Gain alone cannot bypass E floor')
    rows[-1]['treatment_prediction']='E';r=reduce_v238(rows)
    need(r['gate']['YCB_DEV_E_recall']==.5 and r['gate']['numeric_gate_pass'],'Exact50percent boundary passes floor')
    need(metrics(rows[:2],'control')['CM_D_E_by_D_E_U']==[[1,1,0],[0,0,0]],'2x3 metric math')
    need(r['DEV_paired']['YCB_AUX']['E']['recovered']==1 and r['gate']['current_admission'] is False,'Paired E recovery not model adoption')
    need(r['SYN_q_reference_distance']['DEV']['LE3']['control']['N']==4 and len(metric_table(r))==20,'Closed3px strata and table schema')
    try:entry_authority({'approved_completed_SOURCE_saved_summary':False})
    except ValueError:pass
    else:raise AssertionError('False actual entry rejected')
    print(json.dumps(dict(completed=True,PASS=True,checks=6,actual_results_models_features_or_payloads_read=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--entry',type=Path);parser.add_argument('--self-test',action='store_true');parser.add_argument('--verify-saved',action='store_true');a=parser.parse_args()
    if a.self_test:self_test()
    elif a.verify_saved:verify_saved()
    else:need(a.entry is not None,'--entry required');run(a.entry)
