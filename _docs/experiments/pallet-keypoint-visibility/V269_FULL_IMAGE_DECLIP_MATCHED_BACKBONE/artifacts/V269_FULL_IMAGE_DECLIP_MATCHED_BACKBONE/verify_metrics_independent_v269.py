#!/usr/bin/env python3
"""Independent arithmetic from sealed JSON/CSV; no evaluator/model execution."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
import os
import stat

HERE=Path(__file__).absolute().parent
ARMS=('EVA_STANDARD','EVA_CSA','EVA_DECLIP')
MODELS=(*ARMS,'OPENAI_FULL_CLIP','V267_CS','V264_HEATMAP_LOCAL','CAL_LEGACY')


def safe(value,fresh=False):
    p=Path(value)
    if not p.is_absolute() or '..' in p.parts or any(x in str(p).casefold() for x in ('v112','cache','.pyc')):
        raise ValueError(str(p))
    for a in [*reversed(p.parents),p]:
        try:mode=os.lstat(a).st_mode
        except FileNotFoundError:continue
        if stat.S_ISLNK(mode):raise ValueError(str(a))
    if fresh and p.exists():raise FileExistsError(str(p))
    return p


def records(path):
    return [json.loads(x) for x in safe(path).read_bytes().splitlines()]


def near(r):
    try:
        return r['human_state']=='DIRECT_VISIBLE' and r['annotation_coordinate_source']=='manual_click' and r['original_query_valid']=='True' and float(r['initial_reference_error_px'])<=3
    except (ValueError,TypeError):return False


def ratio(a,b):
    return a/b if b else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',default='evaluation_v1')
    parser.add_argument('--out',default='ROOT_INDEPENDENT_METRICS_CHECK.json')
    args=parser.parse_args()
    assert Path(args.evaluation).name==args.evaluation and Path(args.out).name==args.out
    freeze=json.loads(safe(HERE/'CURRENT_PREDICTIONS_FROZEN.json').read_bytes())
    assert freeze['completed'] and freeze.get('PASS',True)
    root=safe(HERE/args.evaluation);inp=safe(root/'compact_inputs')
    integrity=json.loads(safe(inp/'INTEGRITY.json').read_bytes())
    for name,b in integrity['files'].items():
        assert Path(name).name==name
        data=safe(inp/name).read_bytes()
        assert len(data)==b['size_bytes'] and hashlib.sha256(data).hexdigest()==b['sha256']
    assert integrity['files']['NEW_PREDICTIONS.jsonl']['sha256']==freeze['predictions']['sha256']
    assert integrity['files']['LEDGER.csv']['sha256']=='df7b619649ede26af4e21644c22b10e84b334c6bc8708f4e5a668ec373db5175'
    with safe(inp/'LEDGER.csv').open() as f:ledger=list(csv.DictReader(f))
    new=records(inp/'NEW_PREDICTIONS.jsonl')
    prior=records(inp/'V264_PRIOR_PREDICTIONS.jsonl')
    ref=records(inp/'V267_REFERENCE_PREDICTIONS.jsonl')
    with safe(root/'CLASSIFICATION.csv').open() as f:reported=list(csv.DictReader(f))
    assert len(reported)==84 and len(ledger)==len(new)==len(prior)==len(ref)==1784
    predictions={a:[r[a]['pred'] for r in new] for a in ARMS}
    predictions.update(OPENAI_FULL_CLIP=[r['FULL_CLIP']['pred'] for r in ref],
        V267_CS=[r['FULL_CLIP_SURGERY_ARCH']['pred'] for r in ref],
        V264_HEATMAP_LOCAL=[r['HEATMAP_LOCAL']['pred'] for r in prior],
        CAL_LEGACY=[r['legacy_D_E_U'] for r in ledger])
    for i,r in enumerate(ledger):
        assert int(r['global_index'])==new[i]['global_index']==prior[i]['global_index']==ref[i]['global_index']==i
        assert r['row_id']==new[i]['row_id']==prior[i]['row_id']==ref[i]['row_id']
        assert (r['original_query_valid']=='True')==new[i]['available']
        for arm in ARMS:
            p=new[i][arm]
            if not new[i]['available']:assert p['pred']=='U';continue
            z=float(p['E_logit'])
            assert p['pred']==('E' if z>0 else 'D')
            assert abs(float(p['p_E'])-1/(1+math.exp(-max(-700,min(700,z)))))<2e-6
            assert abs(float(p['p_target'])+float(p['p_E'])-1)<2e-6
    checks=[]
    for r in reported:
        assert r['model'] in MODELS
        selected=[]
        for z in ledger:
            if r['group']!='ALL' and z['group']!=r['group']:continue
            d=z['human_state']=='DIRECT_VISIBLE';de=z['human_state'] in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED')
            ok={'PRIMARY':de and z['paired_same_query_RGB_D_vs_E']=='True','ALL_REGISTERED_D_E':de,
                'MANUAL_CLICK_D_ALL':d and z['annotation_coordinate_source']=='manual_click',
                'MANUAL_CLICK_D_LE3_VALID_INPUT':near(z)}[r['cohort']]
            if ok:selected.append(z)
        c=Counter(('D' if z['human_state']=='DIRECT_VISIBLE' else 'E',predictions[r['model']][int(z['global_index'])]) for z in selected)
        dd,de,du,ed,ee,eu=[c[k] for k in [('D','D'),('D','E'),('D','U'),('E','D'),('E','E'),('E','U')]]
        integers=dict(N=len(selected),truth_D=dd+de+du,truth_E=ed+ee+eu,D_D=dd,D_E=de,D_U=du,E_D=ed,E_E=ee,E_U=eu,U=du+eu)
        for k,v in integers.items():assert int(r[k])==v,(r['model'],r['cohort'],k)
        values=dict(D_precision=ratio(dd,dd+ed),D_recall=ratio(dd,dd+de+du),D_F1=ratio(2*dd,2*dd+ed+de+du),
            E_precision=ratio(ee,ee+de),E_recall=ratio(ee,ee+ed+eu),E_F1=ratio(2*ee,2*ee+de+ed+eu),
            U_rate=ratio(du+eu,len(selected)),coverage=ratio(dd+de+ed+ee,len(selected)),
            accuracy_including_U_as_unresolved=ratio(dd+ee,len(selected)),
            accepted_accuracy=ratio(dd+ee,dd+de+ed+ee),
            balanced_recall=((dd/(dd+de+du)+ee/(ee+ed+eu))/2 if dd+de+du and ee+ed+eu else None))
        for k,v in values.items():assert (v is None and r[k]=='') or (v is not None and abs(v-float(r[k]))<1e-12),(r['model'],r['cohort'],k)
        checks.append(dict(group=r['group'],cohort=r['cohort'],model=r['model'],N=len(selected),CM=[[dd,de,du],[ed,ee,eu]]))
    with safe(root/'SELF_RESPONSES.csv').open() as f:self_report=list(csv.DictReader(f))
    assert len(self_report)==21
    for r in self_report:
        ids=[i for i,z in enumerate(ledger) if z['human_state']=='SELF_OCCLUDED' and (r['group']=='ALL' or z['group']==r['group'])]
        c=Counter(predictions[r['model']][i] for i in ids)
        assert int(r['N'])==len(ids)
        for state in 'DEU':assert int(r['response_'+state])==c[state]
    with safe(root/'EXTERNAL90_TRANSITION_SUMMARY.csv').open() as f:ext_report=list(csv.DictReader(f))
    for r in ext_report:
        ids=[i for i,z in enumerate(ledger) if z['paired_same_query_RGB_D_vs_E']=='True' and z['human_state']=='EXTERNAL_OCCLUDED' and (r['group']=='ALL' or z['group']==r['group'])]
        old=predictions[r['baseline']];newp=predictions[r['model']]
        values=dict(N=len(ids),prior_misses=sum(old[i]=='D' for i in ids),
            RECOVERED_PRIOR_MISS=sum(old[i]=='D' and newp[i]=='E' for i in ids),
            REMAINED_PRIOR_MISS=sum(old[i]=='D' and newp[i]=='D' for i in ids),
            KEPT_PRIOR_CORRECT=sum(old[i]=='E' and newp[i]=='E' for i in ids),
            REGRESSED_PRIOR_CORRECT=sum(old[i]=='E' and newp[i]=='D' for i in ids))
        for k,v in values.items():assert int(r[k])==v,(r['baseline'],r['model'],k)
    result=dict(completed=True,PASS=True,independent_counts=True,
        checked_metric_rows=84,checked_SELF_rows=21,checked_external_transition_rows=len(ext_report),
        checked_rows=checks,model_or_RGB_or_external_GTsource_reads=0,new_optimizer_or_PnP=0,
        scope='Sealed prediction JSON and copied late-evaluation ledger only; no inherited evaluator code executed.')
    safe(HERE/args.out,True).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(PASS=True,checked_metric_rows=84,checked_SELF_rows=21,checked_external_transition_rows=len(ext_report))))


if __name__=='__main__':main()
