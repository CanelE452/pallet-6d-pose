"""All1904 predictions first, fixed-reference eight-arm paired evaluation."""
from pathlib import Path
import argparse
import csv
from collections import Counter
from scripts.research.pallet_minimal_bi_ablation_v1 import batch as B
from scripts.research.pallet_minimal_bi_ablation_v1 import runtime as RT
from scripts.research.pallet_minimal_bi_ablation_v1 import scoring as OLD
from scripts.research.pallet_minimal_bi_ablation_v1.statistics import (
    METRICS,distribution,PairedClusterBootstrap,transitions,P_VALUE_REASON)
from .run import ARMS,verify

def summarize(rows,contrasts):
    boot=PairedClusterBootstrap(rows);arms={};comparisons={}
    for a in ARMS:
        m=[r['metrics'][a] for r in rows];status=[r['prediction_status'][a] for r in rows]
        arms[a]=dict(denominator=len(rows),successes_le5=sum(x['success_le5'] is True for x in m),
            successes_strict_lt5=sum(x['success_strict_lt5'] is True for x in m),
            no_pose_n=sum(not x['pose_available'] for x in m),unresolved_n=sum(x['success_le5'] is None for x in m),
            exact5_n=sum(x['translation_cm']==5. or x['rotation_deg']==5. for x in m),
            accepted_n=sum(x['accepted'] is True for x in status),fallback_n=sum(x['fallback'] is True for x in status),
            acceptance_known_n=sum(x['accepted'] is not None for x in status),
            errors={metric:distribution(x[metric] for x in m) for metric in METRICS},
            status_counts=dict(Counter(x['status'] for x in status)))
    for a,b in contrasts:
        diff=[None if any(r['metrics'][x]['success_le5'] is None for x in (a,b)) else
            float(r['metrics'][a]['success_le5'])-float(r['metrics'][b]['success_le5']) for r in rows]
        sr=boot.contrast(diff);sr['gains']=sr.pop('worsened');sr['losses']=sr.pop('improved')
        sr['unit']='success-rate difference; multiply by100 for percentage points'
        comparisons[a+'-'+b]=dict(new=a,reference=b,transition=transitions(rows,a,b),success_rate=sr,
            continuous={metric:boot.contrast([None if any(r['metrics'][x][metric] is None for x in (a,b)) else
                r['metrics'][a][metric]-r['metrics'][b][metric] for r in rows]) for metric in METRICS},
            p_raw=None,p_Holm=None,p_value_reason=P_VALUE_REASON)
    return dict(rows=len(rows),arms=arms,comparisons=comparisons,bootstrap=boot.metadata())

def gate(folder):
    folder=Path(folder);p=verify(folder);seal=B.read(folder/'PREDICTION_SEAL.json')
    if (seal['rows']!=238 or seal['arm_rows']!=1904 or seal['arms']!=list(ARMS) or
        seal['protocol_sha256']!=RT.sha256(folder/'PROTOCOL.json') or
        seal['predictions_sha256']!=RT.sha256(folder/'PREDICTIONS.jsonl.gz') or
        seal['observation_seal_sha256']!=RT.sha256(folder/'run/OBSERVATION_SEAL.json') or
        seal['independent_bindings_sha256']!=RT.sha256(folder/'INDEPENDENT_BINDINGS.json') or
        not seal['before_new_scoring']):raise ValueError('Complete fixed prediction seal required')
    rows=list(B.rows(folder/'PREDICTIONS.jsonl.gz'));frames=B.read(folder/'inputs_v1/INPUTS.json')['frames']
    if len(rows)!=238 or [r['id'] for r in rows]!=p['ordered_ids'] or len(set(p['ordered_ids']))!=238:
        raise ValueError('Ordered fixed238 required')
    if dict(Counter(r['partition'] for r in rows))!={'CAL':46,'TEST':49,'REMAINDER':143}:raise ValueError('Fixed populations required')
    for r,f in zip(rows,frames):
        if any(r[k]!=f[k] for k in ('id','partition','session','cluster_id')) or set(r['predictions'])!=set(ARMS):raise ValueError('Pairing changed')
        for a in ARMS:OLD.normalize_entry(r['predictions'][a],a)
        for arm,branch in (('N3_F','N3_ONLY'),('SUBPIX_F','SUBPIX_ONLY')):
            pose=dict(f['arms'][branch]['coarse_pose']);pose.setdefault('available',True)
            if r['predictions'][arm]['actual_pose']!=pose:raise ValueError('Own baseline pose changed')
    return rows,p

def score(folder,prior,source_root):
    folder,prior=Path(folder),Path(prior);rows,p=gate(folder);output=folder/'score'
    if output.exists():raise FileExistsError('Preserve every scoring attempt')
    old=list(B.rows(prior/'run_v2/PREDICTIONS.jsonl.gz'))
    for row,before in zip(rows,old):
        if row['id']!=before['id'] or any(row['predictions'][a]!=before['predictions'][a] for a in ('R0','R1','N0','N1')):
            raise ValueError('Existing four predictions changed')
    manifest=B.read(prior/'SCORING_REFERENCE_MANIFEST_V2.json')
    refs,ann=OLD.read_fixed_references(source_root=source_root,manifest=manifest,compact=rows)
    scored=[]
    for r in rows:
        scored.append(dict(id=r['id'],partition=r['partition'],session=r['session'],cluster_id=r['cluster_id'],
            metrics={a:OLD.metric_for_pose(r['predictions'][a]['actual_pose'],refs[r['partition']][r['id']],
                remainder=r['partition']=='REMAINDER') for a in ARMS},
            prediction_status={a:{k:r['predictions'][a].get(k) for k in ('accepted','fallback','status','reason')} for a in ARMS}))
    original=B.read(prior/'score_v2/SCORED_ROWS.json')['rows']
    for r,oldrow in zip(scored,original):
        if r['id']!=oldrow['id'] or any(r['metrics'][a]!=oldrow['metrics'][a] for a in ('R0','R1','N0','N1')):
            raise ValueError('Old four-arm score reproduction changed')
    pops={part:[r for r in scored if r['partition']==part] for part in ('CAL','TEST','REMAINDER')}
    pops['PRIMARY']=pops['TEST']+pops['REMAINDER'];results={k:summarize(v,p['contrasts']) for k,v in pops.items()}
    B.write(output/'SCORED_ROWS.json',dict(rows=scored,arm_rows=1904))
    B.write(output/'RESULTS.json',dict(schema='SINGLE_COMPONENT_FULL_FIXED238_EIGHT_ARM_SCORE_V1',
        populations=results,no_method_selection=True,all_rows_kept=True,default_changed=False,
        old_four_arm_poses_and_metrics_exact=True,success='T<=5cm AND R<=5deg',independent_test=False))
    flat=[dict(id=r['id'],partition=r['partition'],session=r['session'],cluster_id=r['cluster_id'],arm=a,
        **r['metrics'][a],**r['prediction_status'][a]) for r in scored for a in ARMS]
    with (output/'ROW_METRICS.csv').open('x',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    changed=[dict(population=pop,contrast=c,category=cat,id=fid) for pop,summary in results.items()
        for c,value in summary['comparisons'].items() for cat in ('failure_to_success','success_to_failure')
        for fid in value['transition']['ids'][cat]]
    with (output/'CHANGED_CASES.csv').open('x',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=['population','contrast','category','id']);w.writeheader();w.writerows(changed)
    B.write(output/'SCORE_SEAL.json',dict(rows=238,arm_rows=1904,prediction_seal_sha256=RT.sha256(folder/'PREDICTION_SEAL.json'),
        reference_bindings=manifest['reference_bindings'],annotation_sha256=ann,
        artifacts={x.name:RT.sha256(x) for x in output.iterdir() if x.is_file()},
        source_sha256={Path(__file__).name:RT.sha256(__file__)},all_rows_kept=True))
    for arm in ARMS:
        s=results['PRIMARY']['arms'][arm]
        print(arm,s['successes_le5'],'/192','meanT',round(s['errors']['translation_cm']['mean'],6),
            'maxT',round(s['errors']['translation_cm']['max'],6),flush=True)
    print('SCORE_SEAL',RT.sha256(output/'SCORE_SEAL.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True);a=p.parse_args();score(a.folder,a.prior,a.source_root)
