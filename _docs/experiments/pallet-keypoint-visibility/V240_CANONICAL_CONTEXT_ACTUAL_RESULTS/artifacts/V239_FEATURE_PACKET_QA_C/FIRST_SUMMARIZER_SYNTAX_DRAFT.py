"""양 final1000 동결 후 저장 CSV만 SOURCE 산술/보고. --verify-saved는 표준 라이브러리만."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import traceback
from bound_io import need, identity, safe, verify, read_raw, load_json, binding, write_json, open_entry
from source_metrics_math import ARMS, reduce_canonical_source, metric_table

COLS = ['sample_index','row_id','family','role','available','target','logit0','logit1',
        'selected_family_logit','prediction']

def flag(value):
    need(value in ('True','False'), 'CSV boolean literal')
    return value == 'True'

def parse_rows(raw):
    reader = csv.DictReader(io.StringIO(raw.decode()))
    need(reader.fieldnames == COLS, '고정 CSV schema')
    rows = []
    for i,r in enumerate(reader):
        for k in ('sample_index','family','target'): r[k] = int(r[k])
        r['available'] = flag(r['available'])
        need(r['sample_index'] == i and r['family'] in (0,1) and r['target'] in (0,1)
             and r['role'] in ('FIT','DEV'), '원 sample order/target/family')
        for k in ('logit0','logit1','selected_family_logit'):
            r[k] = None if r[k] == '' else float(r[k])
        if r['available']:
            need(all(r[k] is not None and math.isfinite(r[k]) for k in
                ('logit0','logit1','selected_family_logit')), 'available finite logits')
            z = r['logit'+str(r['family'])]
            need(z == r['selected_family_logit'] and r['prediction'] == ('E' if z >= 0 else 'D'), '고정 logit0 decode')
        else:
            need(r['prediction'] == 'U' and all(r[k] is None for k in
                ('logit0','logit1','selected_family_logit')), '원 unavailable만 U')
        rows.append(r)
    need(len({r['row_id'] for r in rows}) == len(rows), '단일 row identity')
    return rows

def join(control,treat,original):
    need(len(control) == len(treat) == 7617 and len(original) == 15040, '원 known/full 분모')
    meta = {}; groups = {}; masked = 0
    for r in original:
        gi,fit = int(r['group_index']),bool(r['fit'])
        need(gi not in groups or groups[gi] == fit, '원 FIT/DEV group 겹침 금지'); groups[gi] = fit
        if r['sample_index'] is None:
            need(int(r['target']) == -1, '7423 masked 유지'); masked += 1; continue
        i = int(r['sample_index']); need(i not in meta, '단일 sample mapping');meta[i] = r
    need(masked == 7423 and set(meta) == set(range(7617)), '모든 known/masked 보존')
    rows=[]
    for i,(a,b) in enumerate(zip(control,treat)):
        for k in ('sample_index','row_id','family','role','available','target'):
            need(a[k] == b[k], '두 final arm 원 metadata 동일')
        r=meta[i]
        need(a['row_id'] == r['row_id'] and a['family'] == int(r['family_code'])
             and a['target'] == int(r['target']) and a['role'] == ('FIT' if r['fit'] else 'DEV')
             and a['available'] == bool(r['crop_available']), '원 metadata exact join')
        rows.append(dict(sample_index=i,row_id=a['row_id'],family=a['family'],role=a['role'],
            target=a['target'],available=a['available'],group_index=int(r['group_index']),
            q_reference_distance_px=None,control_prediction=a['prediction'],treatment_prediction=b['prediction'],
            control_logit=a['selected_family_logit'],treatment_logit=b['selected_family_logit']))
    need(sum(r['role']=='FIT' for r in rows)==5876 and sum(r['role']=='DEV' for r in rows)==1741
         and sum(r['family']==0 for r in rows)==6096 and sum(r['family']==1 for r in rows)==1521, '원 SOURCE counts')
    return rows

def summary_agrees(summary, producer):
    names={'SYN_MAIN':0,'YCB_AUX':1}
    for name,f in names.items():
        for role in ('FIT','DEV'):
            for key,arm in zip(('control','treatment'),ARMS):
                ours=summary['family_metrics'][name][role][key];theirs=producer['arms'][arm][f'family{f}_{role}']
                need(ours['N']==theirs['rows'] and ours['CM_D_E_by_D_E_U']==theirs['CM_D_E_by_D_E_U'], 'producer CM 검산')
                for state in ('D','E'): need(ours[state]==theirs[state], 'producer class P/R 검산')
    g=summary['gate']; p=producer['SOURCE_gate']
    need(g['numeric_gate_pass']==p['PASS'] and abs(g['macro_gain_pp']-p['mean_family_DEV_balanced_recall_gain_pp'])<1e-10
         and abs(-min(g['class_recall_deltas'])*100-p['max_family_class_recall_drop_pp'])<1e-10,
         '고정 gate 산술 검산')

def markdown(summary):
    lines=['# V239 canonical SOURCE 두 팔 저장 결과','',
        '같은 공통 특징으로 CONTROL 1,000회와 TREAT 1,000회를 각각 새로 학습한 저장 출력의 산술입니다. '
        'SYN은 intended-native witness0 MAIN, YCB는 associated-r AUX이며 서로 다른 참조 대상을 하나의 과업으로 합치지 않습니다.',
        '', '| 가족 | 역할 | 팔 | N | D recall | E recall | D precision | E precision | U |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    def pct(v):return 'UNKNOWN' if v is None else f'{100*v:.2f}%'
    for f,roles in summary['family_metrics'].items():
        for role,arms in roles.items():
            for arm,m in arms.items():
                lines.append(f"| {f} | {role} | {arm} | {m['N']} | {pct(m['D']['recall'])} | {pct(m['E']['recall'])} | {pct(m['D']['precision'])} | {pct(m['E']['precision'])} | {m['U']} |")
    g=summary['gate'];lines += ['',f"고정 SOURCE gate: {'PASS' if g['numeric_gate_pass'] else 'FAIL'}. 평균 DEV balanced recall 변화 {g['macro_gain_pp']:.4f}pp. '
        '기준은 +3pp / 각 family·class 하락 최대 2pp / YCB E recall 50% / SYN D·E 각각 90%입니다.',
        '', '원 15,040행 중 알려진 D/E 7,617행만 이 출력 분모이며 masked 7,423행은 모델 U가 아닙니다. '
        'S/U 문맥 노드는 원 입력 지원에 따라 보존됩니다. q-reference 거리 sidecar를 이 wrapper가 읽지 않으므로 거리별 표는 UNKNOWN으로 남깁니다.',
        '', 'SOURCE DEV는 이미 노출·재사용된 자료입니다. 추가 residual 9,956개 파라미터의 효과를 완전한 동일용량 인과 실험으로 해석하지 않습니다. '
        '현재/held 평가·PnP는 새로 수행하지 않았고 current admission=false, 선택 모델 없음입니다.',
        '', '[전체 CM·P/R·게이트](SUMMARY.json) · [수치표](METRICS.csv) · [행별 paired 출력](ALL7617_PAIRED_ROWS.jsonl)', '']
    return '\n'.join(lines)

def run(entry_path):
    e,_=open_entry(entry_path,'approved_saved_SOURCE_pair_report',__file__)
    verify(e['metrics_code'],e['code_whitelist'])
    need(e['metrics_code']['sha256']=='d8eef669713163e93ba1883c5784c4a94a6a348cceb06c41f7937f01c98d1afc'
         and identity(e['metrics_code'])==identity(binding(Path(__file__).parent/'source_metrics_math.py')), '봉인된 원 산술 byte pin')
    meta=e['metadata_whitelist'];c=load_json(e['fit_complete'],meta)
    need(c['completed'] is True and c['PASS'] is True and c['code']['sha256']==
         'ce3380b5938132717d349b7be27545e5347110da8663d2b698ec44c96dbfe419', '완료 canonical trainer pin')
    need(c['new_optimizer_updates']==dict(zip(ARMS,[1000,1000])) and c['new_DEV_head_output_before_bothfinalfreeze']==0
         and c['new_V235_CF']==0 and c['current_admission'] is False, '양1000/noCF/no current')
    freeze=load_json(c['freeze'],meta);qa=load_json(c['canonical_feature_QA'],meta)
    need(freeze['completed'] is True and freeze['PASS'] is True and freeze['all_final_updates']==dict(zip(ARMS,[1000,1000]))
         and freeze['new_DEV_head_output_before_bothfinalfreeze']==0 and freeze['old_CONTROL_or_prefix_used'] is False
         and freeze['checkpoints']==c['checkpoints'] and qa['completed'] is True and qa['PASS'] is True
         and qa['checked']==7617 and qa['active_first384_byte_exact'] is True, '양final/global QA 장벽')
    need(identity(e['freeze'])==identity(c['freeze']) and identity(e['canonical_feature_QA'])==identity(c['canonical_feature_QA']), 'entry 장벽 binding')
    need(set(e['outputs'])==set(ARMS) and all(identity(e['outputs'][a])==identity(c['outputs'][a]) for a in ARMS), '두 CSV closure')
    original=[json.loads(s) for s in read_raw(verify(e['sample_map'],meta)).decode().splitlines()]
    need(e['sample_map']['sha256']=='82d4ca430e038e6567538e7dc373069f621913a480fc3a69c17d32634a1e12c6', '원 map pin')
    data=[parse_rows(read_raw(verify(e['outputs'][a],meta))) for a in ARMS]
    rows=join(*data,original);s=reduce_canonical_source(rows);summary_agrees(s,load_json(c['summary'],meta))
    out=safe(e['output_directory']);need(not out.exists(),'fresh report output');out.mkdir()
    try:
        write_json(out/'SUMMARY.json',s)
        with (out/'ALL7617_PAIRED_ROWS.jsonl').open('x') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        table=metric_table(s)
        with (out/'METRICS.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
        with (out/'PORTABLE_REPORT_KO.md').open('x') as f:f.write(markdown(s))
        files=[binding(out/name) for name in ('SUMMARY.json','ALL7617_PAIRED_ROWS.jsonl','METRICS.csv','PORTABLE_REPORT_KO.md')]
        manifest=write_json(out/'MANIFEST.json',dict(completed=True,PASS=True,files=[dict(relative_path=Path(x['path']).name,
            sha256=x['sha256'],size_bytes=x['size_bytes']) for x in files]))
        receipt=write_json(out/'COMPLETE.json',dict(completed=True,PASS=True,fit_complete=e['fit_complete'],freeze=c['freeze'],
            entry=binding(entry_path),code=binding(__file__),manifest=manifest,counts=dict(original=15040,known=7617,masked=7423,FIT=5876,DEV=1741),
            current_admission=False,selected_arm=None,arrays_weights_RGB_or_protected_reads=0))
        print(json.dumps(receipt))
    except BaseException:
        write_json(out/'FAILED_EXECUTION.json',dict(completed=False,PASS=False,traceback=traceback.format_exc()));raise

def verify_saved(directory):
    out=safe(directory);m=json.loads(read_raw(out/'MANIFEST.json'));names=set()
    for b in m['files']:
        name=b['relative_path'];need(Path(name).name==name and name not in names,'saved own basename');names.add(name)
        own=dict(path=str(out/name),sha256=b['sha256'],size_bytes=b['size_bytes']);verify(own,[own])
    need(names=={'SUMMARY.json','ALL7617_PAIRED_ROWS.jsonl','METRICS.csv','PORTABLE_REPORT_KO.md'},'saved exact scalar leaves')
    rows=[json.loads(x) for x in read_raw(out/'ALL7617_PAIRED_ROWS.jsonl').decode().splitlines()]
    summary=reduce_canonical_source(rows);need(summary==json.loads(read_raw(out/'SUMMARY.json')),'saved reducer replay')
    expected=io.StringIO(newline='');table=metric_table(summary);w=csv.DictWriter(expected,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    need(read_raw(out/'METRICS.csv')==expected.getvalue().encode(),'saved metric table replay')
    need(read_raw(out/'PORTABLE_REPORT_KO.md')==markdown(summary).encode(),'saved report replay')
    print(json.dumps(dict(completed=True,PASS=True,stdlib_saved_arithmetic_only=True,rows=len(rows),gate=summary['gate'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--entry');g.add_argument('--verify-saved');a=p.parse_args()
    run(a.entry) if a.entry else verify_saved(a.verify_saved)
