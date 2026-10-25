"""확정 SOURCE 두팔 saved-output 산술만. I/O/모델/학습 선택 없음."""
import math
from source_metrics_math import reduce_canonical_source,metric_table as _metric_table

ARMS=('BASELINE_V239_CANONICAL_CONTEXT','TREAT_SCENE_CF_MATCHED_DATA')

def decode_row(r,available):
    z=[r['logit0'],r['logit1']];f=r['family']
    if not available:
        if z!=[None,None] or r['selected_family_logit'] is not None or r['prediction']!='U':raise ValueError('Input unavailable is explicit U')
    else:
        if f not in (0,1) or any(not isinstance(v,(float,int)) or not math.isfinite(v) for v in z) or r['selected_family_logit']!=z[f] or r['prediction']!=('E' if z[f]>=0 else 'D'):raise ValueError('Fixed selected-family scalar zero threshold')

def natural_pair(control,treatment):
    if len(control)!=7617 or len(treatment)!=7617:raise ValueError('Same original7617, not CF augmentation denominator')
    rows=[]
    for i,(a,b) in enumerate(zip(control,treatment)):
        if any(a[k]!=b[k] for k in ('sample_index','row_id','family','role','available','target')) or a['sample_index']!=i:raise ValueError('Original paired row identity/role/target')
        if a['family'] not in (0,1) or a['role'] not in ('FIT','DEV') or a['target'] not in (0,1):raise ValueError('Separate known natural source referents')
        decode_row(a,a['available']);decode_row(b,b['available'])
        rows.append(dict(row_id=a['row_id'],family=a['family'],role=a['role'],available=a['available'],target=a['target'],control_prediction=a['prediction'],treatment_prediction=b['prediction'],q_reference_distance_px=None))
    result=reduce_canonical_source(rows)
    # Frozen math emits the old two-new-arm metadata. Numeric gates stay identical.
    result.update(arms=list(ARMS),new_CONTROL_optimizer_updates=0,new_TREAT_optimizer_updates=1000,original_rows15040_retained=True,masked7423_not_in_known7617=True,CF_diagnostics_are_separate=True)
    result.pop('SYN_q_reference_distance',None) # no unapproved q/r sidecar reading or inference
    return result

def metric_table(summary):
    return _metric_table({**summary,'SYN_q_reference_distance':{}})

def cf_diagnostics(rows):
    if len(rows)!=10812:raise ValueError('All5406 parents x2 CF roles, no failure filtering')
    seen=set();out={}
    for r in rows:
        key=r['pair_index'],r['variant']
        if key in seen or r['variant'] not in ('COVER','ADJACENT_SHAM') or r['role']!='FIT' or r['native_gold'] is not False:raise ValueError('Original FIT pair/role/weak authority')
        seen.add(key)
        y=(1 if r['variant']=='COVER' else 0) if r['proof_PASS'] else -1
        if r['target']!=y or (r['proof_PASS'] and not r['constructed']):raise ValueError('Input availability and target authority separate')
        decode_row(r,r['constructed'])
    if seen!={(i,v) for i in range(5406) for v in ('COVER','ADJACENT_SHAM')}:raise ValueError('Full CF pair census')
    for f in (0,1):
        take=[r for r in rows if r['family']==f];known=[r for r in take if r['target']>=0]
        cm=[[sum(r['target']==y and r['prediction']==p for r in known) for p in ('D','E','U')] for y in (0,1)]
        out['family'+str(f)]={'CM_D_E_by_D_E_U':cm,'known_rows':len(known),'UNKNOWN_rows':len(take)-len(known)}
    return out
