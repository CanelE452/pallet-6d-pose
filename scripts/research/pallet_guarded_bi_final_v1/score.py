"""CAL-only fixed selection, then complete locked PRIMARY evaluation.

Inference never imports this module. Original fixed physical/C2 metrics and
reference byte bindings are reused without modifying any label or alignment.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
import argparse
import csv
from scripts.research.pallet_minimal_bi_ablation_v1 import batch as B
from scripts.research.pallet_minimal_bi_ablation_v1 import runtime as RT
from scripts.research.pallet_minimal_bi_ablation_v1 import scoring as OLD
from scripts.research.pallet_minimal_bi_ablation_v1.statistics import (
    METRICS, distribution, transitions, PairedClusterBootstrap, P_VALUE_REASON)
from .run import ARMS,verify

SELECTION_RULE=dict(candidate_modes=['0','A','B'],
    per_initialization='A/B eligible only if CAL own-baseline damage=0 and recovery>=1; otherwise own baseline',
    per_initialization_sort=['joint_success_desc','own_damage_asc','translation_mean_asc','rotation_mean_asc','simple_mode_0_A_B'],
    global_sort=['CAL_joint_success_desc','initialization_priority_E_N_R'],
    no_PRIMARY_selection=True,no_ID_or_partition_routing=True)

def gate(folder):
    folder=Path(folder);p=verify(folder);seal=B.read(folder/'PREDICTION_SEAL.json')
    if (seal['predictions_sha256']!=RT.sha256(folder/'PREDICTIONS.jsonl.gz') or
        seal['guard_audits_sha256']!=RT.sha256(folder/'GUARD_AUDITS.jsonl.gz') or
        seal['independent_bindings_sha256']!=RT.sha256(folder/'INDEPENDENT_BINDINGS.json') or
        seal['protocol_sha256']!=RT.sha256(folder/'METHOD_PROTOCOL.json') or
        seal['rows']!=238 or seal['arm_rows']!=2856 or seal['arms']!=list(ARMS) or
        seal['before_new_CAL_scoring'] is not True): raise ValueError('Full prediction seal required')
    rows=list(B.rows(folder/'PREDICTIONS.jsonl.gz'))
    if [r['id'] for r in rows]!=seal['ordered_ids'] or len(set(seal['ordered_ids']))!=238:
        raise ValueError('Ordered238 identities required')
    if dict(Counter(r['partition'] for r in rows))!={'CAL':46,'TEST':49,'REMAINDER':143}:
        raise ValueError('Full fixed populations required')
    for r in rows:
        if set(r['predictions'])!=set(ARMS): raise ValueError('All12 arms required')
        for a in ARMS: OLD.normalize_entry(r['predictions'][a],a)
    return rows,p

def score_rows(rows,refs):
    out=[]
    for r in rows:
        out.append(dict(id=r['id'],partition=r['partition'],session=r['session'],cluster_id=r['cluster_id'],
            metrics={a:OLD.metric_for_pose(r['predictions'][a]['actual_pose'],refs[r['partition']][r['id']],
                remainder=r['partition']=='REMAINDER') for a in ARMS},
            prediction_status={a:{k:r['predictions'][a].get(k) for k in ('accepted','fallback','status','reason')} for a in ARMS}))
    return out

def summarize(rows,contrasts=()):
    summaries={}
    for a in ARMS:
        ms=[r['metrics'][a] for r in rows];ss=[r['prediction_status'][a] for r in rows]
        summaries[a]=dict(denominator=len(rows),successes_le5=sum(m['success_le5'] is True for m in ms),
            successes_strict_lt5=sum(m['success_strict_lt5'] is True for m in ms),
            no_pose_n=sum(not m['pose_available'] for m in ms),
            unresolved_n=sum(m['success_le5'] is None for m in ms),
            accepted_n=sum(s['accepted'] is True for s in ss),
            fallback_n=sum(s['fallback'] is True for s in ss),
            acceptance_known_n=sum(s['accepted'] is not None for s in ss),
            exact5_n=sum(m['translation_cm']==5. or m['rotation_deg']==5. for m in ms),
            errors={m:distribution(x[m] for x in ms) for m in METRICS},
            status_counts=dict(Counter(s['status'] for s in ss)))
    comparisons={};boot=PairedClusterBootstrap(rows)
    for a,b in contrasts:
        binary=[None if r['metrics'][a]['success_le5'] is None or r['metrics'][b]['success_le5'] is None else
            float(r['metrics'][a]['success_le5'])-float(r['metrics'][b]['success_le5']) for r in rows]
        success=boot.contrast(binary)
        success['gains']=success.pop('worsened');success['losses']=success.pop('improved')
        success['unit']='success-rate difference; multiply by100 for percentage points'
        comparisons[a+'-'+b]=dict(transition=transitions(rows,a,b),success_rate=success,
            continuous={m:boot.contrast([None if r['metrics'][a][m] is None or r['metrics'][b][m] is None else
                r['metrics'][a][m]-r['metrics'][b][m] for r in rows]) for m in METRICS},
            p_raw=None,p_Holm=None,p_value_reason=P_VALUE_REASON)
    return dict(rows=len(rows),arms=summaries,comparisons=comparisons,bootstrap=boot.metadata())

def selection(scored):
    if len(scored)!=46 or any(r['partition']!='CAL' for r in scored):raise ValueError('Only CAL46 may select')
    stats=summarize(scored);by_init={};eligibility={}
    for prefix in ('R','N','E'):
        eligible=[prefix+'0']
        for mode in ('A','B'):
            arm=prefix+mode;tr=transitions(scored,arm,prefix+'0')['counts']
            ok=tr['success_to_failure']==0 and tr['failure_to_success']>=1
            eligibility[arm]=dict(eligible=ok,recovery=tr['failure_to_success'],damage=tr['success_to_failure'])
            if ok:eligible.append(arm)
        def key(a):
            d=stats['arms'][a]
            return (-d['successes_le5'],eligibility.get(a,{}).get('damage',0),
                d['errors']['translation_cm']['mean'],d['errors']['rotation_deg']['mean'],'0AB'.index(a[1]))
        by_init[prefix]=min(eligible,key=key)
    chosen=min(by_init.values(),key=lambda a:(-stats['arms'][a]['successes_le5'],'ENR'.index(a[0])))
    return dict(selected_arm=chosen,per_initialization=by_init,eligibility=eligibility,selection_rule=SELECTION_RULE)

def cal(folder,source_root,prior):
    folder,prior=Path(folder),Path(prior);rows,p=gate(folder)
    manifest=B.read(prior/'SCORING_REFERENCE_MANIFEST_V2.json')
    binding=manifest['reference_bindings']['CAL'];path=Path(source_root)/binding['path']
    if RT.sha256(path)!=binding['sha256']:raise ValueError('Original CAL reference changed')
    requested=[r for r in rows if r['partition']=='CAL'];labels=B.read(path)
    refs={'CAL':{r['id']:labels['rows'][r['id']]['target']['truth'] for r in requested}}
    scored=score_rows(requested,refs);output=folder/'cal'
    B.write(output/'SCORED_ROWS.json',dict(rows=scored,arm_rows=552))
    B.write(output/'RESULTS.json',summarize(scored))
    result=selection(scored)
    result.update(UTC=datetime.now(timezone.utc).isoformat(),ordered_CAL_ids=[r['id'] for r in scored],
        CAL_scored_rows_sha256=RT.sha256(output/'SCORED_ROWS.json'),
        CAL_reference_sha256=binding['sha256'],prediction_seal_sha256=RT.sha256(folder/'PREDICTION_SEAL.json'),
        method_protocol_sha256=RT.sha256(folder/'METHOD_PROTOCOL.json'),
        guard_source_sha256=RT.sha256(Path(__file__).with_name('guard.py')),
        score_source_sha256=RT.sha256(__file__),before_new_PRIMARY_scoring=True,
        source_hashes=p['inference_source_sha256'],PRIMARY_used_for_selection=False,
        previously_exposed_development_data=True)
    B.write(folder/'SELECTION_SEAL.json',result)
    print('CAL_SELECTION',result['selected_arm'],result['per_initialization'],flush=True)

def primary(folder,source_root,prior):
    folder,prior=Path(folder),Path(prior);rows,p=gate(folder)
    sealed=B.read(folder/'SELECTION_SEAL.json');calrows=B.read(folder/'cal/SCORED_ROWS.json')['rows']
    if sealed['CAL_scored_rows_sha256']!=RT.sha256(folder/'cal/SCORED_ROWS.json') or any(
        sealed[k]!=v for k,v in selection(calrows).items()) or not sealed['before_new_PRIMARY_scoring']:
        raise ValueError('CAL choice must be sealed before PRIMARY')
    if sealed['prediction_seal_sha256']!=RT.sha256(folder/'PREDICTION_SEAL.json'):raise ValueError('Predictions changed')
    manifest=B.read(prior/'SCORING_REFERENCE_MANIFEST_V2.json')
    refs,annotations=OLD.read_fixed_references(source_root=source_root,manifest=manifest,compact=rows)
    scored=score_rows(rows,refs);out=folder/'score';contrasts=[]
    for prefix in ('R','N','E'):
        contrasts.extend((prefix+m,prefix+'0') for m in ('1','A','B'))
    chosen=sealed['selected_arm']
    contrasts+= [(chosen,a) for a in ('E0','N0','N1','R0','R1') if a!=chosen]
    contrasts=list(dict.fromkeys(contrasts))
    pops={part:[r for r in scored if r['partition']==part] for part in ('CAL','TEST','REMAINDER')}
    pops['PRIMARY']=pops['TEST']+pops['REMAINDER']
    results={pop:summarize(values,contrasts) for pop,values in pops.items()}
    B.write(out/'SCORED_ROWS.json',dict(rows=scored,arm_rows=2856))
    B.write(out/'RESULTS.json',dict(populations=results,selected_arm=chosen,default_changed=False,
        selection_seal_sha256=RT.sha256(folder/'SELECTION_SEAL.json'),
        independent_validation=False,all_rows_and_failures_kept=True,success='T<=5cm AND R<=5deg'))
    flat=[dict(id=r['id'],partition=r['partition'],session=r['session'],cluster_id=r['cluster_id'],arm=a,
        **r['metrics'][a],**r['prediction_status'][a]) for r in scored for a in ARMS]
    out.mkdir(parents=True,exist_ok=True)
    with (out/'ROW_METRICS.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    B.write(out/'SCORE_SEAL.json',dict(rows=238,arm_rows=2856,all_rows_kept=True,
        pre_PRIMARY_selection_seal_sha256=RT.sha256(folder/'SELECTION_SEAL.json'),
        prediction_seal_sha256=RT.sha256(folder/'PREDICTION_SEAL.json'),reference_bindings=manifest['reference_bindings'],
        annotation_sha256=annotations,source_sha256={Path(__file__).name:RT.sha256(__file__)},
        artifacts={x.name:RT.sha256(x) for x in out.iterdir() if x.is_file()}))
    for a in ARMS:
        s=results['PRIMARY']['arms'][a]
        print(a,s['successes_le5'],'/192','meanT',round(s['errors']['translation_cm']['mean'],4),
            'maxT',round(s['errors']['translation_cm']['max'],4),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('cal','primary'))
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True);a=p.parse_args()
    (cal if a.stage=='cal' else primary)(a.folder,a.source_root,a.prior)
if __name__=='__main__':main()
