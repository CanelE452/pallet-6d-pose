"""Saved SOURCE 결과만: Root entry 후 실행. 모델/특징/배열/영상 decoder 없음."""
import argparse,csv,hashlib,html,io,json,math,os
from pathlib import Path
import metadata_entry_math as meta
from report_math import ARMS,natural_pair,cf_diagnostics,metric_table

H=Path(__file__).absolute().parent
NATURAL=('sample_index','row_id','family','role','available','target','logit0','logit1','selected_family_logit','prediction')
CF=('pair_index','parent_sample_index','row_id','family','role','variant','constructed','proof_PASS','target','logit0','logit1','selected_family_logit','prediction','native_gold')
INTS={'sample_index','pair_index','parent_sample_index','family','target'}
BOOLS={'available','constructed','proof_PASS','native_gold'}
FLOATS={'logit0','logit1','selected_family_logit'}

def boolean(s):meta.need(s in ('True','False'),'CSV bool');return s=='True'
def typed(raw,columns):
    rr=list(csv.DictReader(io.StringIO(raw.decode(),newline='')))
    meta.need(bool(rr) and set(rr[0])==set(columns),'fixed CSV columns')
    for r in rr:
        for k in r:
            if k in INTS:r[k]=int(r[k])
            elif k in BOOLS:r[k]=boolean(r[k])
            elif k in FLOATS:
                r[k]=None if r[k]=='' else float(r[k]);meta.need(r[k] is None or math.isfinite(r[k]),'finite or explicit null scalar')
    return rr

def read_csv(b,white,columns):
    meta.need(meta.identity(b) in {meta.identity(x) for x in white} and Path(b['path']).suffix=='.csv','exact saved CSV whitelist before FS')
    raw=meta.raw(b['path']);meta.need(hashlib.sha256(raw).hexdigest()==b['sha256'] and ('size_bytes' not in b or len(raw)==b['size_bytes']),'saved CSV SHA/size')
    return typed(raw,columns),raw

def compare(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(compare(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(compare(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(int,float)) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
    return a==b

def check_natural(control,treatment):
    s=natural_pair(control,treatment)
    meta.need(len({r['row_id'] for r in control})==7617 and sum(r['role']=='FIT' for r in control)==5876 and sum(r['role']=='DEV' for r in control)==1741,'all7617 natural identity/role counts')
    meta.need(sum(r['family']==0 for r in control)==6096 and sum(r['family']==1 for r in control)==1521 and all(r['available'] for r in control),'original separated known families/support')
    return s

def check_CF(rows,natural):
    m=cf_diagnostics(rows);pairs={}
    for i,r in enumerate(rows):
        meta.need((r['variant'],r['pair_index'])==(('COVER' if i<5406 else 'ADJACENT_SHAM'),i%5406),'all original ordered5406 x2 roles')
        p=r['parent_sample_index'];meta.need(0<=p<7617,'CF original parent bounds');n=natural[p]
        meta.need(n['target']==0 and n['role']=='FIT' and n['available'] and n['row_id']==r['row_id'] and n['family']==r['family'],'same original FIT D parent identity')
        item=(p,r['row_id'],r['family'],r['constructed'],r['proof_PASS'])
        if r['pair_index'] in pairs:meta.need(pairs[r['pair_index']]==item,'paired sameparent availability/label authority')
        else:pairs[r['pair_index']]=item
    meta.need(len(pairs)==5406 and len({v[0] for v in pairs.values()})==5406 and sum(v[3] for v in pairs.values())==5367 and sum(v[4] for v in pairs.values())==5367,'all5406/constructed5367/unknown39, not success subset')
    return m

def producer_compare(result,control,treatment,cf,producer):
    for a,short,rows in zip(ARMS,('control','treatment'),(control,treatment)):
        expected={}
        for f,name in ((0,'SYN_MAIN'),(1,'YCB_AUX')):
            for role in ('FIT','DEV'):
                m=result['family_metrics'][name][role][short]
                expected[f'family{f}_{role}']=dict(rows=m['N'],known_DE=m['N'],CM_D_E_by_D_E_U=m['CM_D_E_by_D_E_U'],target_UNKNOWN=0,original_input_U=sum(not r['available'] for r in rows if r['family']==f and r['role']==role),D=m['D'],E=m['E'])
        meta.need(compare(expected,producer['arms'][a]),'producer natural CM/P/R independent match')
    changes={}
    for f,name in ((0,'SYN_MAIN'),(1,'YCB_AUX')):
        for role in ('FIT','DEV'):
            take=[(a,b) for a,b in zip(control,treatment) if a['family']==f and a['role']==role]
            gain=sum(a['prediction']!=('D','E')[a['target']] and b['prediction']==('D','E')[b['target']] for a,b in take)
            harm=sum(a['prediction']==('D','E')[a['target']] and b['prediction']!=('D','E')[b['target']] for a,b in take)
            changes[f'family{f}_{role}']=dict(rows=len(take),recovered=gain,damaged=harm,net=gain-harm)
    meta.need(compare(changes,producer['paired_changes']) and compare(cf,producer['CF_FIT_only']),'producer paired/CF independent match')
    meta.need(producer['all_original_rows']==15040 and producer['known_output_rows']==7617 and producer['masked_target_minus1_retained']==7423 and producer['current_admission'] is False,'original masked/full denominators')
    # The producer uses the same gate with descriptive field names. Counts above
    # independently define all recalls; audit the producer gate after mapping.
    pg=producer['SOURCE_gate'];g=result['gate']
    meta.need(pg['PASS']==g['numeric_gate_pass'],'fixed SOURCE gate PASS arithmetic')
    expected_gate={'mean_family_DEV_balanced_recall_gain_pp':g['macro_gain_pp'],'max_family_class_recall_drop_pp':-100*min(g['class_recall_deltas']),'YCB_DEV_E_recall':g['YCB_DEV_E_recall'],'SYN_DEV_D_recall':g['SYN_DEV_D_recall'],'SYN_DEV_E_recall':g['SYN_DEV_E_recall'],'fixed_min_gain_pp':3,'fixed_max_class_drop_pp':2,'fixed_YCB_DEV_E_floor':.5,'fixed_SYN_DEV_D_E_floor':.9,'current_admission':False,'automatic_retry':False,'new_current_inference':0}
    meta.need(all(k in pg and compare(v,pg[k]) for k,v in expected_gate.items()),'producer fixed gate scalars and floors')
    return changes

def jsave(p,o):
    with os.fdopen(os.open(meta.safe(p),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return meta.binding(p)

def csvsave(p,rows):
    with os.fdopen(os.open(meta.safe(p),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return meta.binding(p)

def merged(control,treatment):
    out=[]
    for a,b in zip(control,treatment):
        row={k:a[k] for k in NATURAL[:6]}
        row.update({'control_'+k:a[k] for k in NATURAL[6:]});row.update({'treatment_'+k:b[k] for k in NATURAL[6:]});out.append(row)
    return out

def table_text(table):
    cols=('family','role','arm','N','DD','DE','DU','ED','EE','EU','D_precision','D_recall','E_precision','E_recall','U_rate','balanced_recall')
    def cell(v):return '—' if v is None else f'{100*v:.2f}%' if isinstance(v,float) else str(v)
    return cols,[[cell(r[k]) for k in cols] for r in table]

def run(entry):
    e=json.loads(meta.raw(entry));meta.need(e.get('approved_saved_SOURCE_scene_CF_report') is True and e.get('source_only') is True and e.get('automatic_retry') is False,'Root saved-report entry required before any result access')
    codewhite=e['code_whitelist'];meta.need(meta.identity(e['code'])==meta.identity(meta.binding(__file__)),'exact driver CODE')
    for b in e['code_files']:meta.checked(b,codewhite)
    white=e['metadata_whitelist'];scope=meta.checked(e['root_scope'],white);meta.need(scope.get('approved_saved_SOURCE_scene_CF_report') is True,'separate saved-output scope')
    ready=meta.report_config(e,white)
    fitqa=meta.checked(e['fit_QA'],white);meta.passed(fitqa);meta.same(fitqa['fit_complete'],e['fit_complete'])
    meta.need(e['expected_source_plan_sha256']=='50e547b4c07afdee690727d895278ad7b82fe80eab52551dd00b9f5eb7a5d0e3','fixed prefeature source-plan SHA')
    c=meta.checked(e['fit_complete'],white)
    meta.need(c['draw_ledger']['sha256']==e['expected_actual_draw_ledger_sha256']=='b2c1c3bff32981b2570f0330353c3c4f54fcfa4fb41d209b0dacfccfc1ac0dae','whole prefeature prospective ledger byte identity')
    for arm in ARMS:meta.same(e['natural_CSVs'][arm],ready['natural_CSVs'][arm])
    meta.same(e['CF_FIT_CSV'],ready['CF_FIT_CSV']);meta.same(e['producer_summary'],ready['producer_summary'])
    wanted=[*e['natural_CSVs'].values(),e['CF_FIT_CSV']];meta.need({meta.identity(x) for x in e['CSV_whitelist']}=={meta.identity(x) for x in wanted},'only exact3 saved CSVs')
    control,_=read_csv(e['natural_CSVs'][ARMS[0]],e['CSV_whitelist'],NATURAL);treatment,_=read_csv(e['natural_CSVs'][ARMS[1]],e['CSV_whitelist'],NATURAL)
    cf,cfraw=read_csv(e['CF_FIT_CSV'],e['CSV_whitelist'],CF);producer=meta.checked(e['producer_summary'],white);meta.passed(producer)
    result=check_natural(control,treatment);cfmetrics=check_CF(cf,control);changes=producer_compare(result,control,treatment,cfmetrics,producer)
    result.update(completed=True,PASS=True,fit_complete=e['fit_complete'],freeze=c['freeze'],fit_QA=e['fit_QA'],CF_FIT_only=cfmetrics,producer_paired_changes=changes,producer_summary_binding=e['producer_summary'],producer_arithmetic_match=True,current_admission=False,selected_arm=None,model_adoption=False,new_current_inference=0)
    out=meta.safe(e['output_directory']);meta.need(out.parent==H and not out.exists(),'fresh own results folder');out.mkdir()
    artifacts=[];artifacts.append(csvsave(out/'PAIRED_NATURAL7617.csv',merged(control,treatment)))
    with os.fdopen(os.open(out/'CF_DIAGNOSTIC10812.csv',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(cfraw)
    artifacts.append(meta.binding(out/'CF_DIAGNOSTIC10812.csv'));artifacts.append(jsave(out/'SUMMARY.json',result))
    table=metric_table(result);artifacts.append(csvsave(out/'NATURAL_METRICS.csv',table));cols,body=table_text(table)
    gate=result['gate'];headline='PASS' if gate['numeric_gate_pass'] else 'FAIL'
    note=f'고정 SOURCE gate {headline}. 새 treatment1000과 과거 baseline byte재사용0newupdates의 제한된 SOURCE 비교입니다. SOURCE DEV는 노출·재사용됐으며 새 blind 검증이나 현재 팔레트 실사 성공이 아닙니다.'
    md=['# V245 동일 장면 CF: 저장 SOURCE 결과','',note,'','SYN intended-native witness0와 YCB associated-r AUX(native UNKNOWN)를 별도 표로 유지합니다. 자연 known7617과 CF FIT10812를 합치지 않으며, 원15040 중 masked7423은 보존됩니다. 문턱은각family scalar0입니다.','','| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    md.extend('| '+' | '.join(row)+' |' for row in body)
    md+=['','고정 gate는 family DEV balanced recall 평균 +3pp 이상·각 family/class recall 감소 2pp 이하·YCB E50% 이상·SYN D/E 각각90% 이상입니다. CF target UNKNOWN은 CM 밖에 남기며 input unavailable U와 구분합니다. CF 성적은 이미 노출된 FIT 진단으로 자연 DEV 일반화 성적이 아닙니다. q/L/native/U 또는 라벨을 수정하지 않았습니다.','','[전체 수치](SUMMARY.json) · [metric CSV](NATURAL_METRICS.csv) · [paired 자연행](PAIRED_NATURAL7617.csv) · [CF 진단행](CF_DIAGNOSTIC10812.csv) · [HTML](PORTABLE_REPORT_KO.html)','']
    p=out/'PORTABLE_REPORT_KO.md';p.write_text('\n'.join(md));artifacts.append(meta.binding(p))
    h='<html lang="ko"><meta charset="utf-8"><title>V245 SOURCE saved result</title><style>body{font:16px sans-serif;margin:24px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:6px;text-align:right}.scroll{overflow-x:auto}</style><h1>V245 동일 장면 CF SOURCE 결과</h1><p>'+html.escape(note)+'</p><p>SYN intended-native witness0 / YCB associated-r AUX(native UNKNOWN). 자연7617과 CF10812는 별도 분모이며 원 masked7423은 유지됩니다. 모델 채택·현재셋 개선 주장이 아닙니다.</p><div class="scroll"><table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in body)+'</table></div><p><a href="SUMMARY.json">JSON</a> · <a href="NATURAL_METRICS.csv">CSV</a> · <a href="PORTABLE_REPORT_KO.md">Markdown</a></p></html>'
    p=out/'PORTABLE_REPORT_KO.html';p.write_text(h);artifacts.append(meta.binding(p))
    manifest=jsave(out/'MANIFEST.json',{'files':[{'path':Path(b['path']).name,'sha256':b['sha256'],'size_bytes':b['size_bytes']} for b in artifacts]})
    complete=jsave(out/'COMPLETE.json',dict(completed=True,PASS=True,fit_complete=e['fit_complete'],freeze=c['freeze'],fit_QA=e['fit_QA'],summary=meta.binding(out/'SUMMARY.json'),metric_table=meta.binding(out/'NATURAL_METRICS.csv'),report=meta.binding(out/'PORTABLE_REPORT_KO.md'),portable_HTML=meta.binding(out/'PORTABLE_REPORT_KO.html'),manifest=manifest,code=e['code'],source_gate_numeric_PASS=gate['numeric_gate_pass'],current_admission=False,selected_arm=None,optimizer_updates_executed_by_report=0,model_or_threshold_selected_by_report=False,raw_arrays_model_features_RGB_GT_current_or_held_reads=0))
    print(json.dumps({'COMPLETE':complete,'SOURCE_gate':gate},ensure_ascii=False),flush=True)

def verify_saved(directory):
    out=meta.safe(directory);manifest=json.loads(meta.raw(out/'MANIFEST.json'))
    for b in manifest['files']:
        meta.need(Path(b['path']).name==b['path'],'owned local leaf only');raw=meta.raw(out/b['path']);meta.need(hashlib.sha256(raw).hexdigest()==b['sha256'] and len(raw)==b['size_bytes'],'saved output identity')
    rr=list(csv.DictReader(io.StringIO(meta.raw(out/'PAIRED_NATURAL7617.csv').decode(),newline='')));arms=[[],[]]
    for r in rr:
        for i,a in enumerate(('control','treatment')):
            x={k:r[k] for k in NATURAL[:6]};x.update({k:r[a+'_'+k] for k in NATURAL[6:]})
            raw=io.StringIO();w=csv.DictWriter(raw,fieldnames=NATURAL);w.writeheader();w.writerow(x);arms[i].extend(typed(raw.getvalue().encode(),NATURAL))
    s=check_natural(*arms);cf=typed(meta.raw(out/'CF_DIAGNOSTIC10812.csv'),CF);cfm=check_CF(cf,arms[0]);stored=json.loads(meta.raw(out/'SUMMARY.json'))
    for k in ('family_metrics','DEV_paired','gate'):meta.need(compare(s[k],stored[k]),'saved arithmetic '+k)
    meta.need(compare(cfm,stored['CF_FIT_only']),'saved CF arithmetic');print(json.dumps(dict(completed=True,PASS=True,stdlib_verify_saved=True,natural_rows=7617,CF_rows=10812,current_admission=False)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--entry');g.add_argument('--verify-saved');a=p.parse_args()
    if a.entry:run(a.entry)
    else:verify_saved(a.verify_saved)
