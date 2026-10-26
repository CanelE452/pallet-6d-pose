"""Independent direct counts, without executing either inherited evaluator."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

HERE=Path(__file__).absolute().parent


def safe(p):
    p=Path(p)
    assert p.is_absolute()and not any(x in str(p).lower()for x in ('v112','cache','.pyc','..'))
    for a in reversed((p,*p.parents)):assert not a.is_symlink()
    return p


def records(p):return [json.loads(x)for x in safe(p).read_text().splitlines()]
def near(r):
    try:return r['human_state']=='DIRECT_VISIBLE'and r['annotation_coordinate_source']=='manual_click'and r['original_query_valid']=='True'and float(r['initial_reference_error_px'])<=3
    except ValueError:return False


def main():
    inp=HERE/'evaluation_v2/compact_inputs'
    integrity=json.loads(safe(inp/'INTEGRITY.json').read_text())
    for name,b in integrity['files'].items():
        raw=safe(inp/name).read_bytes();assert len(raw)==b['size_bytes']and hashlib.sha256(raw).hexdigest()==b['sha256']
    ledger=list(csv.DictReader(safe(inp/'LEDGER.csv').open()))
    new=records(inp/'NEW_PREDICTIONS.jsonl');old=records(inp/'V264_PRIOR_PREDICTIONS.jsonl')
    reported=list(csv.DictReader(safe(HERE/'evaluation_v2/CLASSIFICATION.csv').open()))
    predictions={arm:[r[arm]['pred']for r in new]for arm in ('FULL_CLIP','FULL_CLIP_SURGERY_ARCH')}
    predictions['V264_HEATMAP_LOCAL']=[r['HEATMAP_LOCAL']['pred']for r in old]
    predictions['v61']=[r['v61_D_E_U']for r in ledger];predictions['CAL_LEGACY']=[r['legacy_D_E_U']for r in ledger]
    for i,r in enumerate(ledger):assert int(r['global_index'])==i and r['row_id']==new[i]['row_id']==old[i]['row_id']
    checks=[]
    for r in reported:
        selected=[]
        for z in ledger:
            if r['group']!='ALL'and z['group']!=r['group']:continue
            d=z['human_state']=='DIRECT_VISIBLE';de=z['human_state']in('DIRECT_VISIBLE','EXTERNAL_OCCLUDED')
            ok={'PRIMARY':de and z['paired_same_query_RGB_D_vs_E']=='True','ALL_REGISTERED_D_E':de,
                'MANUAL_CLICK_D_ALL':d and z['annotation_coordinate_source']=='manual_click',
                'MANUAL_CLICK_D_LE3_VALID_INPUT':near(z)}[r['cohort']]
            if ok:selected.append(z)
        counts=Counter(('D'if z['human_state']=='DIRECT_VISIBLE'else'E',predictions[r['model']][int(z['global_index'])])for z in selected)
        for a in 'DE':
            for b in 'DEU':assert counts[a,b]==int(r[a+'_'+b])
        dd,de,du,ed,ee,eu=[counts[k]for k in [('D','D'),('D','E'),('D','U'),('E','D'),('E','E'),('E','U')]]
        def ratio(a,b):return a/b if b else None
        vals=dict(D_precision=ratio(dd,dd+ed),D_recall=ratio(dd,dd+de+du),E_precision=ratio(ee,ee+de),E_recall=ratio(ee,ee+ed+eu),U_rate=ratio(du+eu,len(selected)))
        for k,v in vals.items():assert (v is None and r[k]=='')or(v is not None and abs(v-float(r[k]))<1e-12)
        checks.append(dict(group=r['group'],cohort=r['cohort'],model=r['model'],N=len(selected),CM=[[dd,de,du],[ed,ee,eu]]))
    external=[int(r['global_index'])for r in ledger if r['paired_same_query_RGB_D_vs_E']=='True'and r['human_state']=='EXTERNAL_OCCLUDED']
    previous=predictions['V264_HEATMAP_LOCAL'];miss=[i for i in external if previous[i]=='D'];assert len(miss)==67
    transitions={}
    for arm in ('FULL_CLIP','FULL_CLIP_SURGERY_ARCH'):
        p=predictions[arm];transitions[arm]=dict(recovered=[i for i in miss if p[i]=='E'],remain=[i for i in miss if p[i]=='D'],regressed=[i for i in external if previous[i]=='E'and p[i]=='D'])
    assert len(transitions['FULL_CLIP_SURGERY_ARCH']['recovered'])==61 and len(transitions['FULL_CLIP_SURGERY_ARCH']['regressed'])==0
    result=dict(completed=True,PASS=True,independent_counts=True,checked_rows=checks,external_transitions=transitions,
        model_RGB_or_GTsource_reads=0,optimizer_updates=0,source='Sealedpredictions+copiedlate-evaluationledger, no evaluator code executed.')
    with safe(HERE/'ROOT_INDEPENDENT_METRICS_CHECK.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(PASS=True,checked_metric_rows=len(checks),transition_counts={a:{k:len(v)for k,v in z.items()}for a,z in transitions.items()})))


if __name__=='__main__':main()
