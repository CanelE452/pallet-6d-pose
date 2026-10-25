#!/usr/bin/env python3
"""One saved-CSV arithmetic check; no model/forward/original payload reads."""
import csv, hashlib, io, json, math, os, stat
from pathlib import Path
import numpy as np
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=R/'V233_TRAIN_COMPLETION_QA_B'
ENTRY=H/'ROOT_ACTUAL_COMPLETION_QA_ENTRY.json'
RUN=R/'V233_ROOT_CROP_ADAPTATION/common_local_crop_pair_v2'
OUT=H/'actual_review_v1'
ARMS=('CONTROL_LOCAL_CROP','TREAT_LAST_QKV_LORA4')
access=[]
def safe(p):
    p=Path(p)
    assert p.is_absolute() and '..' not in p.parts
    low=str(p).lower(); assert 'v112' not in low and '__pycache__' not in low
    assert p.is_relative_to(R)
    for q in [p]+list(p.parents):
        assert not q.is_symlink(), str(q)
    return p
entry=json.loads(safe(ENTRY).read_text())
# Bindings originate solely from the sealed explicit actual QA entry whitelist.
white={b['path']:b for b in entry['whitelist']}
def read(p):
    p=safe(p); b=white[str(p)]
    with p.open('rb') as f: raw=f.read()
    assert hashlib.sha256(raw).hexdigest()==b['sha256']
    if 'size_bytes' in b: assert len(raw)==b['size_bytes']
    access.append({'path':str(p),'sha256':b['sha256'],'size_bytes':len(raw)})
    return raw
summary=json.loads(read(RUN/'SOURCE_RESULTS.json'))
samplepath=R/'V233_ROOT_CROP_ADAPTATION/crop_samples_v1/samples.npz'
with np.load(io.BytesIO(read(samplepath)),allow_pickle=False) as z:
    sample={k:z[k] for k in ('row_id','family','fit','available','target')}
n=len(sample['row_id']); assert n==7617 and all(len(v)==n for v in sample.values())
assert len(set(map(str,sample['row_id'])))==n
all_rows={}; metrics={}
def near(a,b):
    if isinstance(a,float): assert b is not None and math.isclose(a,b,rel_tol=0,abs_tol=1e-12),(a,b)
    else: assert a==b,(a,b)
for arm in ARMS:
    rows=list(csv.DictReader(io.StringIO(read(RUN/(arm+'_ALL_SOURCE_ROWS.csv')).decode())))
    assert len(rows)==n
    groups={f'family{f}_{role}':[] for f in (0,1) for role in ('FIT','DEV')}
    for i,r in enumerate(rows):
        f=int(r['family']); target=int(r['target']); avail=r['available']=='True'
        role='FIT' if bool(sample['fit'][i]) else 'DEV'
        assert int(r['sample_index'])==i and r['row_id']==str(sample['row_id'][i])
        assert f==int(sample['family'][i]) and target==int(sample['target'][i])
        assert avail==bool(sample['available'][i]) and r['role']==role
        vals=[float(r['logit0']),float(r['logit1']),float(r['selected_family_logit'])]
        assert all(math.isfinite(v) for v in vals)
        assert vals[2]==vals[f]
        pred='U' if not avail else ('E' if vals[2]>=0 else 'D')
        assert pred==r['prediction'] and target in (0,1)
        groups[f'family{f}_{role}'].append((target,pred))
    metrics[arm]={}
    for key,rr in groups.items():
        cm=[[0,0,0],[0,0,0]]
        for target,pred in rr: cm[target][{'D':0,'E':1,'U':2}[pred]]+=1
        m={'rows':len(rr),'known_DE':len(rr),'target_UNKNOWN':0,'original_input_U':0,'CM_D_E_by_D_E_U':cm}
        for k,c in enumerate(('D','E')):
            truth=sum(cm[k]); predicted=sum(row[k] for row in cm); tp=cm[k][k]
            m[c]={'truth':truth,'predicted':predicted,'precision':tp/predicted if predicted else None,'recall':tp/truth if truth else None}
        saved=summary['arms'][arm][key]
        assert set(m)==set(saved)
        for k,v in m.items():
            if isinstance(v,dict):
                assert set(v)==set(saved[k])
                for a,b in v.items():near(b,saved[k][a])
            else:near(v,saved[k])
        metrics[arm][key]=m
    all_rows[arm]=rows
paired={}
for key in metrics[ARMS[0]]:
    f=int(key[6]); role=key[8:]; recovery=damage=changed=stable_correct=stable_wrong=0
    for a,b in zip(all_rows[ARMS[0]],all_rows[ARMS[1]]):
        assert all(a[k]==b[k] for k in ('sample_index','row_id','family','role','available','target'))
        if int(a['family'])!=f or a['role']!=role: continue
        truth='DE'[int(a['target'])]; ac=a['prediction']==truth; bc=b['prediction']==truth
        recovery+=int(not ac and bc); damage+=int(ac and not bc)
        changed+=int(a['prediction']!=b['prediction']); stable_correct+=int(ac and bc); stable_wrong+=int(not ac and not bc)
    paired[key]={'recovery':recovery,'damage':damage,'prediction_changes':changed,'both_correct':stable_correct,'both_wrong':stable_wrong}
assert summary['two_referents_not_merged'] is True
assert summary['no_current_or_practical_transfer_claim'] is True
report={'completed':True,'PASS':True,'scope':'Saved SOURCE CSV arithmetic and row-to-samples join only','source_complete':{'path':str(RUN/'COMPLETE.json'),'sha256':'dd662d0a466e38972a386104f9a26f235f955b060c20da4a3b678283f70ed114'},'rows_per_arm':n,'metrics':metrics,'paired':paired,'DEV_macro_balanced_recall':{arm:sum(metrics[arm][f'family{f}_DEV'][c]['recall'] for f in (0,1) for c in ('D','E'))/4 for arm in ARMS},'access':access,'limits':['No model instantiated or forward/gradient replay.','7617 rows are the available known-D/E sample subset; original15040 row retention and unknown exclusion were separately checked by the saved crop QA.','SYN intended-native witness0 MAIN and YCB associated-r AUX are distinct target authorities; not a human native visibility benchmark.','No current/held/private/original RGB/GT/pretrained/prefix access; no model selection, threshold or recipe change.']}
out=OUT/'METRICS_QA.json';assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
csvout=OUT/'CLASS_PRECISION_RECALL.csv';assert not csvout.exists()
with csvout.open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['arm','group','class','N','truth','predicted','precision','recall','U'])
    for arm,groups in metrics.items():
        for group,m in groups.items():
            for c in ('D','E'):w.writerow([arm,group,c,m['rows'],m[c]['truth'],m[c]['predicted'],m[c]['precision'],m[c]['recall'],sum(r[2] for r in m['CM_D_E_by_D_E_U'])])
print(json.dumps({'completed':True,'PASS':True,'metrics_qa':str(out),'DEV_macro_balanced_recall':report['DEV_macro_balanced_recall'],'paired':paired}))
