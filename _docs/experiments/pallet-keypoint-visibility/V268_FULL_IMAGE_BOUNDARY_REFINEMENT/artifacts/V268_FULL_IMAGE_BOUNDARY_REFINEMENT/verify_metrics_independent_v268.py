"""Independent direct CSV/JSON counts; executes no inherited evaluator/model."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

HERE=Path(__file__).absolute().parent
ARMS=('HIGHRES_REFINE','BOUNDARY_REFINE')


def safe(p):
    p=Path(p)
    assert p.is_absolute()and not any(x in str(p).lower()for x in('v112','cache','.pyc','..'))
    for a in reversed((p,*p.parents)):assert not a.is_symlink()
    return p


def records(p):return[json.loads(x)for x in safe(p).read_text().splitlines()]


def near(r):
    try:return r['human_state']=='DIRECT_VISIBLE'and r['annotation_coordinate_source']=='manual_click'and r['original_query_valid']=='True'and float(r['initial_reference_error_px'])<=3
    except ValueError:return False


def main():
    assert safe(HERE/'CURRENT_PREDICTIONS_FROZEN.json').exists()
    inp=HERE/'evaluation_v1/compact_inputs'
    integrity=json.loads(safe(inp/'INTEGRITY.json').read_text())
    for name,b in integrity['files'].items():
        raw=safe(inp/name).read_bytes();assert len(raw)==b['size_bytes']and hashlib.sha256(raw).hexdigest()==b['sha256']
    with safe(inp/'LEDGER.csv').open()as f:ledger=list(csv.DictReader(f))
    new=records(inp/'NEW_PREDICTIONS.jsonl');old=records(inp/'V264_PRIOR_PREDICTIONS.jsonl');ref=records(inp/'V267_REFERENCE_PREDICTIONS.jsonl')
    with safe(HERE/'evaluation_v1/CLASSIFICATION.csv').open()as f:reported=list(csv.DictReader(f))
    assert len(reported)==72
    predictions={a:[r[a]['pred']for r in new]for a in ARMS}
    predictions.update(V267_CS=[r['FULL_CLIP_SURGERY_ARCH']['pred']for r in ref],
        V264_HEATMAP_LOCAL=[r['HEATMAP_LOCAL']['pred']for r in old],v61=[r['v61_D_E_U']for r in ledger],CAL_LEGACY=[r['legacy_D_E_U']for r in ledger])
    for i,r in enumerate(ledger):assert int(r['global_index'])==i and r['row_id']==new[i]['row_id']==old[i]['row_id']==ref[i]['row_id']
    checks=[]
    for r in reported:
        selected=[]
        for z in ledger:
            if r['group']!='ALL'and z['group']!=r['group']:continue
            d=z['human_state']=='DIRECT_VISIBLE';de=z['human_state']in('DIRECT_VISIBLE','EXTERNAL_OCCLUDED')
            ok={'PRIMARY':de and z['paired_same_query_RGB_D_vs_E']=='True','ALL_REGISTERED_D_E':de,
                'MANUAL_CLICK_D_ALL':d and z['annotation_coordinate_source']=='manual_click','MANUAL_CLICK_D_LE3_VALID_INPUT':near(z)}[r['cohort']]
            if ok:selected.append(z)
        counts=Counter(('D'if z['human_state']=='DIRECT_VISIBLE'else'E',predictions[r['model']][int(z['global_index'])])for z in selected)
        for a in'DE':
            for b in'DEU':assert counts[a,b]==int(r[a+'_'+b])
        dd,de,du,ed,ee,eu=[counts[k]for k in[('D','D'),('D','E'),('D','U'),('E','D'),('E','E'),('E','U')]]
        def ratio(a,b):return a/b if b else None
        vals=dict(D_precision=ratio(dd,dd+ed),D_recall=ratio(dd,dd+de+du),E_precision=ratio(ee,ee+de),E_recall=ratio(ee,ee+ed+eu),U_rate=ratio(du+eu,len(selected)))
        for k,v in vals.items():assert(v is None and r[k]=='')or(v is not None and abs(v-float(r[k]))<1e-12)
        checks.append(dict(group=r['group'],cohort=r['cohort'],model=r['model'],N=len(selected),CM=[[dd,de,du],[ed,ee,eu]]))
    primary=[int(r['global_index'])for r in ledger if r['paired_same_query_RGB_D_vs_E']=='True'and r['human_state']in('DIRECT_VISIBLE','EXTERNAL_OCCLUDED')]
    external=[i for i in primary if ledger[i]['human_state']=='EXTERNAL_OCCLUDED'];visible=[i for i in primary if ledger[i]['human_state']=='DIRECT_VISIBLE']
    assert len(primary)==1237 and len(external)==90 and len(visible)==1147
    transitions={}
    for baseline in('V267_CS','V264_HEATMAP_LOCAL'):
        previous=predictions[baseline];e_miss=[i for i in external if previous[i]=='D'];d_miss=[i for i in visible if previous[i]=='E']
        transitions[baseline]={}
        for arm in ARMS:
            p=predictions[arm]
            transitions[baseline][arm]=dict(external_recovered=[i for i in e_miss if p[i]=='E'],external_remaining=[i for i in e_miss if p[i]=='D'],
                external_new_miss=[i for i in external if previous[i]=='E'and p[i]=='D'],visible_recovered=[i for i in d_miss if p[i]=='D'],
                visible_new_miss=[i for i in visible if previous[i]=='D'and p[i]=='E'])
    result=dict(completed=True,PASS=True,independent_counts=True,checked_metric_rows=len(checks),checked_rows=checks,
        primary_transitions=transitions,model_or_RGB_or_external_GTsource_reads=0,new_optimizer_or_PnP=0,
        source='SealedpredictionJSONandcopiedlate-evaluationledgeronly;noinheritedevaluatorcodeexecuted.')
    with safe(HERE/'ROOT_INDEPENDENT_METRICS_CHECK.json').open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(PASS=True,checked_metric_rows=len(checks),transition_counts={b:{a:{k:len(v)for k,v in z.items()}for a,z in v.items()}for b,v in transitions.items()})))


if __name__=='__main__':main()
