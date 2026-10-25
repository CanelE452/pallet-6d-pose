"""V235 저장 SOURCE 원7617행만 검산한다. 학습 증강10734행은 평가 분모가 아니다."""
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
H=R/'V235_SOURCE_REPORT_C'
CROP=R/'V233_ROOT_CROP_ADAPTATION/crop_samples_v1'
PROTOCOL=dict(path=str(R/'V235_ROOT_PUBLIC_COUNTERFACTUAL/PROTOCOL_BEFORE_DONOR_PIXELS_OR_TREATMENT.json'),sha256='61a354e42b5cf9c85ca3ff554de85750afafaad456f730e307e9cf1e58474880')
PAIRS_COMPLETE=dict(path=str(R/'V235_ROOT_PUBLIC_COUNTERFACTUAL/pairs_v1/COMPLETE.json'),sha256='8958c47fc5cf65d592346d5b9671c592ad470e799cfdcdb926e6e2788ca2fc13')
BASELINE_SUMMARY=dict(path=str(R/'V233_SOURCE_REPORT_C/results/SUMMARY.json'),sha256='8a9b54c59ef94639815e5bd575a48433fc83d52e3776140d16c86c5782546dc4')
ARMS=('CONTROL_LOCAL_CROP','TREAT_CF_DATA_ONLY')
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

def extended_reduction(rows, supplemental=False):
    result=reduce_saved(rows)
    if supplemental:
        result['YCB_supplemental_q_reference_distance']={}
        for role in ('FIT','DEV'):
            result['YCB_supplemental_q_reference_distance'][role]={}
            for label in ('LE3','GT3','UNKNOWN'):
                rr=[r for r in rows if r['family']==1 and r['role']==role and ('UNKNOWN' if r['supplemental_YCB_q_reference_distance_px'] is None else 'LE3' if r['supplemental_YCB_q_reference_distance_px']<=3 else 'GT3')==label]
                result['YCB_supplemental_q_reference_distance'][role][label]={arm:metrics(rr,arm) for arm in ('control','treatment')}
    return result


def extended_table(summary):
    table=metric_table(summary)
    for role,strata in summary.get('YCB_supplemental_q_reference_distance',{}).items():
        for label,arms in strata.items():
            for arm,m in arms.items():
                c=m['CM_D_E_by_D_E_U']
                table.append(dict(family='YCB_AUX',role=role,stratum='SUPPLEMENTAL_'+label,arm=arm,N=m['N'],DD=c[0][0],DE=c[0][1],DU=c[0][2],ED=c[1][0],EE=c[1][1],EU=c[1][2],D_precision=m['D']['precision'],D_recall=m['D']['recall'],E_precision=m['E']['precision'],E_recall=m['E']['recall'],U_rate=m['U_rate'],balanced_recall=m['balanced_recall']))
    return table


def entry_authority(e):
    need(e.get('approved_completed_SOURCE_saved_summary') is True,'Root completed-fit report entry required; no actual CSV access before it')


def supplemental_ycb(binding_or_null,samples):
    if binding_or_null is None:return {}
    rr=list(csv.DictReader(io.StringIO(checked(binding_or_null).decode('utf-8'))))
    need(len(rr)==1521 and len({r['row_id'] for r in rr})==1521,'Supplemental YCB known1521 census')
    by_id={s['row_id']:s for s in samples if s['family']==1};need(len(by_id)==1521,'Original YCB known1521')
    out={}
    for r in rr:
        need(r['row_id'] in by_id,'Supplemental exact original row ID')
        s=by_id[r['row_id']]
        need(int(r['sample_index'])==s['sample_index'] and r['role']==s['role'] and r['associated_r_state']==('D','E')[s['target']],'Supplemental original sample/role/target join')
        distance=float(r['q_reference_distance_px'])
        need(math.isfinite(distance) and distance>=0 and math.isclose(distance,math.hypot(float(r['q_x'])-float(r['r_x_GT_diagnostic_only']),float(r['q_y'])-float(r['r_y_GT_diagnostic_only'])),rel_tol=1e-12,abs_tol=1e-12),'Supplemental exported q/r algebra only')
        out[r['row_id']]=distance
    need(set(out)==set(by_id),'Supplemental full YCB1521 join')
    return out


def report(summary):
    pct=lambda x:'—' if x is None else f'{100*x:.2f}%'
    text=['# V235: 같은 local crop·고정 head에서 SOURCE counterfactual 데이터 대조','',
          '원 SOURCE 7,617행의 저장 결과다. SYN main은 intended-native witness0 associated-point D/E, YCB AUX는 associated-r surface D/E(native UNKNOWN)로 분리한다. 두 referent의 CM을 하나로 합치지 않는다. 원 q/L/native·문턱0은 그대로이며 새 실사 성공·모델 채택을 뜻하지 않는다.','',
          '원 전체 15,040행 중 known D/E 7,617행은 입력 불가도 삭제하지 않고 보존한다. 나머지 7,423행은 원 S/U/제외 target−1로 forward 분모가 아니며 모델 U와 다르다. 학습 증강 COVER/SHAM 행은 별도 FIT counterfactual 자료로서 아래 평가 분모에 더하지 않는다.','',
          '|가족|역할|팔|N|CM [[DD,DE,DU],[ED,EE,EU]]|D P/R|E P/R|U율|균형 recall|',
          '|---|---|---|---:|---|---|---|---|---|']
    for family,roles in summary['family_metrics'].items():
        for role,arms in roles.items():
            for arm,m in arms.items():text.append(f"|{family}|{role}|{arm}|{m['N']}|{m['CM_D_E_by_D_E_U']}|{pct(m['D']['precision'])}/{pct(m['D']['recall'])}|{pct(m['E']['precision'])}/{pct(m['E']['recall'])}|{pct(m['U_rate'])}|{pct(m['balanced_recall'])}|")
    text+=['','|DEV 가족|정답 범위|회복|악화|순증|모두 정답|모두 오답|','|---|---|---:|---:|---:|---:|---:|']
    for family,groups in summary['DEV_paired'].items():
        for label,c in groups.items():text.append(f"|{family}|{label}|{c['recovered']}|{c['damaged']}|{c['net_correct']}|{c['both_correct']}|{c['both_wrong']}|")
    g=summary['gate'];gain='정의불가' if g['macro_gain_pp'] is None else f"{g['macro_gain_pp']:.3f}pp"
    text+=['',f"사전 gate는 두 family DEV balanced recall 평균 +3pp 이상, 각 family/class recall 감소 2pp 이하이다. gain={gain}, 수치 통과={g['numeric_gate_pass']}. 이 계산으로 모델·seed·문턱·데이터를 선택하지 않는다. current admission=false, selected arm=null을 유지한다.",'',
           f"학습 provenance: CONTROL 새 optimizer update={summary['new_CONTROL_optimizer_updates']}, TREAT 새 update={summary['new_TREAT_optimizer_updates']}. CONTROL 원본 재사용 검증은 Root의 봉인 QA binding을 인용한다. 두 checkpoint metadata와 새 head DEV-before-freeze=0을 확인한 뒤에만 저장 CSV를 읽었다. 고정 backbone/같은 head의 data-only 비교이며 이번 trial에 LoRA 실행을 가정하지 않는다. V233의 runtime failure/수리는 역사적 선행 기록이고 V235의 새 실패로 복제하지 않았다.",'',
           f"성공 counterfactual parent={summary['CF_feature_mapping']['successful']}, 실패 masked parent={summary['CF_feature_mapping']['failed_masked']}, COVER/SHAM 학습 증강 행={summary['augmented_training_variant_rows']}. 원 7,617행 평가와 구분한다. 합성 가림은 새 자연 관측 또는 human gold가 아니다.",'',
           'SYN ≤3px/>3px 표는 원 sidecar의 q-reference 거리 진단이다. 원 V233 YCB sidecar의 r와 거리는 계속 null로 보존한다. 별도 승인된 YCB1521 export를 사용했으면 SUPPLEMENTAL strata에만 추가하며, GT 좌표를 모델 입력·q 보정·교사 교체로 사용하지 않는다. q-r 거리는 실제 detector 오차와 같지 않다.','',
           'SOURCE DEV는 이미 노출된 진단이며 fresh-blind가 아니다. 본 검산은 저장 CSV/JSONL과 완료·freeze metadata의 산술/정합 범위다. 원 RGB·GT·NPZ·weights·prefix·모델 forward·신경망 loss 전체를 재실행하지 않았다. 새 current/held 평가·PnP도 없다.','',
           '[수치 JSON](SUMMARY.json) · [CM/P/R/U·거리 strata](METRICS.csv) · [원7617 paired 출력](PAIRED_ALL7617.csv) · [binding·검산 범위](COMPLETE.json)','']
    return '\n'.join(text)


def run(entry_path):
    e=json.loads(raw(entry_path));entry_authority(e)
    need(same(e['code'],binding(Path(__file__).absolute())) and same(e['root_protocol'],PROTOCOL),'Pinned report CODE/root protocol')
    protocol=json.loads(checked(PROTOCOL));gate=protocol['fixed_source_gate']
    need(gate['mean_family_DEV_balanced_recall_gain_min_pp']==3 and gate['max_family_class_D_E_recall_drop_pp']==2 and protocol['model_frozen_design']['threshold_logit']==0,'Prospective fixed numeric gate')
    need(same(e['pairs_complete'],PAIRS_COMPLETE),'Pinned completed pair metadata')
    pairs=json.loads(checked(PAIRS_COMPLETE));need(pairs.get('completed') is True and pairs.get('PASS') is True and same(pairs['protocol'],PROTOCOL),'Completed pairs metadata; payload pointers remain inert')
    need(set(e['inputs'])=={'fit_complete','freeze','crop_complete','sample_diagnostics','full_rows','baseline_summary'},'Closed saved-input set')
    f=json.loads(checked(e['inputs']['fit_complete']))
    need(f.get('completed') is True and f.get('PASS') is True and f['new_CONTROL_optimizer_updates']==0 and f['new_TREAT_optimizer_updates']==1000 and f['new_head_DEV_forward_before_freeze']==0 and f['no_LoRA_or_new_input_channels'] is True,'Actual data-only final1000 gate')
    need(same(f['code'],e['training_code']) and same(f['entry'],e['training_entry']) and same(f['freeze'],e['inputs']['freeze']),'Actual completed code/entry/freeze closure')
    need(set(f['outputs'])==set(ARMS) and set(e['arm_outputs'])==set(ARMS),'Both fixed arms outputs')
    fit=Path(e['inputs']['fit_complete']['path']).parent
    need(fit.is_relative_to(R) and any(p.startswith('V235_') for p in fit.parts),'Explicit V235 own result directory')
    need(e['inputs']['freeze']['path']==str(fit/'FINAL1000_FROZEN_BEFORE_NEW_HEAD_DEV_FORWARD.json'),'Exact new freeze basename')
    for arm in ARMS:need(same(e['arm_outputs'][arm],f['outputs'][arm]) and e['arm_outputs'][arm]['path']==str(fit/(arm+'_ALL_SOURCE_ROWS.csv')),'Own CSV binding '+arm)
    freeze=json.loads(checked(e['inputs']['freeze']))
    need(freeze.get('completed') is True and freeze.get('PASS') is True and freeze['new_head_DEV_forward_before_freeze']==0 and freeze['original_prefix_numeric_rows_before_freeze_only_FIT'] is True,'Both-final before new head DEV')
    need(set(freeze['checkpoints'])==set(ARMS) and freeze['checkpoints']==f['checkpoints'],'Both checkpoint metadata closure; never model bytes')
    for k in ('CONTROL_reuse_QA','CONTROL_replay_QA','draw_source_ledger'):need(same(f[k],e[k]),'Pinned provenance metadata '+k)
    crop=json.loads(checked(e['inputs']['crop_complete']))
    need(e['inputs']['crop_complete']['path']==str(CROP/'COMPLETE.json') and e['inputs']['crop_complete']['sha256']==protocol['original_source']['crop_complete_sha256'],'Unchanged original crop closure')
    need(crop.get('completed') is True and crop.get('PASS') is True and crop['samples']==7617 and crop['all_rows']==15040,'Original sample census')
    cb={x['path']:x for x in crop['outputs']}
    for k,name in [('sample_diagnostics','SAMPLE_DIAGNOSTICS.jsonl'),('full_rows','ALL15040_ROWS.jsonl')]:need(e['inputs'][k]['path']==str(CROP/name) and same(e['inputs'][k],cb[str(CROP/name)]),'Original sidecar pin '+k)
    diag=[json.loads(l) for l in checked(e['inputs']['sample_diagnostics']).splitlines()]
    full=[json.loads(l) for l in checked(e['inputs']['full_rows']).splitlines()]
    need(len(diag)==7617 and len(full)==15040 and len({d['row_id'] for d in full})==15040,'All original SOURCE rows')
    need(Counter((r['family_code'],r['target']) for r in full)=={(0,0):5853,(0,1):243,(0,-1):944,(1,0):1181,(1,1):340,(1,-1):6479},'Original labels untouched')
    by_id={d['row_id']:d for d in full};samples=[]
    for i,d in enumerate(diag):
        need(d==by_id[d['row_id']] and d['sample_index']==i and d['target'] in (0,1) and d['family_code'] in (0,1),'Original sidecar identity')
        dist=d['q_reference_distance_px_diagnostic_only']
        if d['family_code']==1:need(dist is None and d['reference_xy_GT_diagnostic_only'] is None,'Original YCB sidecar null retained')
        elif dist is not None:need(math.isfinite(dist) and dist>=0 and math.isclose(dist,math.dist(d['query_xy'],d['reference_xy_GT_diagnostic_only']),rel_tol=1e-12,abs_tol=1e-12),'Original SYN distance algebra')
        samples.append(dict(sample_index=i,row_id=d['row_id'],family=d['family_code'],target=d['target'],role='FIT' if d['fit'] else 'DEV',available=d['available'],q_reference_distance_px=dist))
    need(Counter((s['family'],s['role']) for s in samples)=={(0,'FIT'):4752,(0,'DEV'):1344,(1,'FIT'):1124,(1,'DEV'):397},'Original primary family/role counts')
    need(sum(d['sample_index'] is None for d in full)==7423 and all(d['target']==-1 for d in full if d['sample_index'] is None),'Masked not forwarded, not prediction U')
    mapping=f['CF_feature_mapping'];need(mapping['successful']+mapping['failed_masked']==5406,'All original FIT D parents retain success/failure')
    allowed_cf=freeze['CF_successful_attempt_indices_only_FIT'];need(len(allowed_cf)==mapping['successful'] and len(set(allowed_cf))==len(allowed_cf),'Successful FIT CF attempt census metadata')
    add=supplemental_ycb(e['supplemental_YCB_reference_diagnostic'],samples)
    arms={arm:parse_arm(e['arm_outputs'][arm],samples) for arm in ARMS};rows=[]
    for i,s in enumerate(samples):
        r=dict(s);r['supplemental_YCB_q_reference_distance_px']=add.get(s['row_id'])
        for arm,short in zip(ARMS,('control','treatment')):
            ar=arms[arm][i];r[short+'_prediction']=ar['prediction'];r[short+'_logit']=None if ar['selected_family_logit']=='' else float(ar['selected_family_logit'])
        r['logit_delta']=None if r['control_logit'] is None else r['treatment_logit']-r['control_logit'];rows.append(r)
    reduction=extended_reduction(rows,bool(add))
    need(same(e['inputs']['baseline_summary'],BASELINE_SUMMARY),'Pinned prior CONTROL summary')
    baseline=json.loads(checked(BASELINE_SUMMARY))
    for fam in ('SYN_MAIN','YCB_AUX'):
        for role in ('FIT','DEV'):need(reduction['family_metrics'][fam][role]['control']==baseline['family_metrics'][fam][role]['control'],'Original CONTROL scalar metrics unchanged')
    summary=dict(completed=True,PASS=True,all_source_rows=15040,known_primary=7617,masked_not_forward=7423,threshold_logit0=True,two_referents_not_merged=True,model_or_threshold_selected=False,new_current_inference=0,
                 new_CONTROL_optimizer_updates=0,new_TREAT_optimizer_updates=1000,CF_feature_mapping=mapping,augmented_training_variant_rows=2*mapping['successful'],original_YCB_reference_distance_still_null=True,supplemental_YCB_distance_enabled=bool(add),**reduction)
    out=safe(e['output_dir']);need(out==H/'results' and not out.exists(),'Fresh report output');out.mkdir()
    with (out/'PAIRED_ALL7617.csv').open('x',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    table=extended_table(summary)
    with (out/'METRICS.csv').open('x',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    json_write(out/'SUMMARY.json',summary)
    with (out/'PORTABLE_REPORT_KO.md').open('x',encoding='utf-8') as file:file.write(report(summary))
    complete=dict(completed=True,PASS=True,code=binding(Path(__file__).absolute()),root_protocol=PROTOCOL,entry=binding(entry_path),inputs=e['inputs'],fit_complete=e['inputs']['fit_complete'],pairs_complete=PAIRS_COMPLETE,arm_outputs=e['arm_outputs'],training_code=e['training_code'],training_entry=e['training_entry'],CONTROL_reuse_QA=e['CONTROL_reuse_QA'],CONTROL_replay_QA=e['CONTROL_replay_QA'],draw_source_ledger=e['draw_source_ledger'],supplemental_YCB_reference_diagnostic=e['supplemental_YCB_reference_diagnostic'],outputs=[binding(out/name) for name in ('SUMMARY.json','METRICS.csv','PAIRED_ALL7617.csv','PORTABLE_REPORT_KO.md')],scope='Saved CSV/JSONL and metadata arithmetic only. No NPZ/model/prefix/RGB/GT or forward/loss replay.',current_admission=False)
    json_write(out/'COMPLETE.json',complete)
    print(json.dumps(dict(completed=True,PASS=True,complete=binding(out/'COMPLETE.json'))))


def verify_saved():
    # package 상대 results만 읽는 stdlib 재검산이며 절대 binding은 따라가지 않는다.
    d=Path(__file__).absolute().parent/'results';summary=json.loads(raw(d/'SUMMARY.json'))
    rows=list(csv.DictReader(io.StringIO(raw(d/'PAIRED_ALL7617.csv').decode())))
    for r in rows:
        for key in ('sample_index','family','target'):r[key]=int(r[key])
        need(r['family'] in (0,1) and r['target'] in (0,1) and r['role'] in ('FIT','DEV'),'Saved known identity')
        r['available']=bool_csv(r['available'])
        for key in ('q_reference_distance_px','supplemental_YCB_q_reference_distance_px'):r[key]=None if r[key]=='' else float(r[key])
        need(r['family']==0 or r['q_reference_distance_px'] is None,'Original YCB null remains distinct')
        logits=[]
        for arm in ('control','treatment'):
            z=None if r[arm+'_logit']=='' else float(r[arm+'_logit']);pred=r[arm+'_prediction']
            need(z is None or math.isfinite(z),'Saved finite logits')
            need(pred==('U' if z is None else 'E' if z>=0 else 'D') and (pred=='U')==(not r['available']),'Saved availability/threshold0')
            logits.append(z)
        need(r['logit_delta']=='' if logits[0] is None else float(r['logit_delta'])==logits[1]-logits[0],'Saved paired logit delta')
    need(len(rows)==7617 and len({r['row_id'] for r in rows})==7617 and [r['sample_index'] for r in rows]==list(range(7617)),'Original primary row order/count')
    actual=extended_reduction(rows,summary['supplemental_YCB_distance_enabled'])
    need(all(summary[k]==v for k,v in actual.items()),'Saved summary exact arithmetic')
    table=list(csv.DictReader(io.StringIO(raw(d/'METRICS.csv').decode())))
    expected=[{k:'' if v is None else str(v) for k,v in row.items()} for row in extended_table(actual)]
    need(table==expected,'All table arithmetic including optional distance strata')
    print(json.dumps(dict(completed=True,PASS=True,rows=7617,scope='stdlib saved output arithmetic only; no original inputs/models')))


def self_test():
    rows=[dict(row_id=str(i),family=f,target=y,role='DEV',available=p!='U',q_reference_distance_px=3 if f==0 else None,supplemental_YCB_q_reference_distance_px=3 if f==1 else None,control_prediction=p,treatment_prediction=t) for i,(f,y,p,t) in enumerate([(0,0,'E','D'),(0,1,'E','E'),(1,0,'D','E'),(1,1,'U','U')])]
    result=extended_reduction(rows,True)
    need(metrics(rows[:2],'control')['CM_D_E_by_D_E_U']==[[0,1,0],[0,1,0]],'Toy 2x3')
    need(result['DEV_paired']['SYN_MAIN']['ALL']['recovered']==1 and result['DEV_paired']['YCB_AUX']['ALL']['damaged']==1,'Toy recovery/damage')
    need(not result['gate']['numeric_gate_pass'] and result['gate']['current_admission'] is False,'Toy fixed gate/no admission')
    need(result['SYN_q_reference_distance']['DEV']['LE3']['control']['N']==2 and result['YCB_supplemental_q_reference_distance']['DEV']['LE3']['control']['N']==2 and rows[2]['q_reference_distance_px'] is None,'Closed3px and supplemental YCB never overwrite original null')
    need(len(extended_table(result))==32,'Both family distance table strata')
    try:entry_authority({'approved_completed_SOURCE_saved_summary':False})
    except ValueError:pass
    else:raise AssertionError('False actual entry rejected')
    print(json.dumps(dict(completed=True,PASS=True,checks=6,actual_results_or_payloads_read=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--entry',type=Path);parser.add_argument('--self-test',action='store_true');parser.add_argument('--verify-saved',action='store_true');args=parser.parse_args()
    if args.self_test:self_test()
    elif args.verify_saved:verify_saved()
    else:need(args.entry is not None,'--entry required');run(args.entry)
