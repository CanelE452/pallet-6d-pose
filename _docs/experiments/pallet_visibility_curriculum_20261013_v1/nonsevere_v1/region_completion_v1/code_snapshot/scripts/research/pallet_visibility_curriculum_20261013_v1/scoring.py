"""Post-seal metrics: proper symmetry, fixed native phase and all-frame counts."""
from collections import Counter, defaultdict
import numpy as np

from . import common as C


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def pose_metric(prediction, truth):
    R, G = np.asarray(prediction['R_physical']), np.asarray(truth['R'])
    t, target_t = np.asarray(prediction['centroid']), np.asarray(truth['t'])
    xyz = C.cuboid(*truth['xyz'])
    angles, adds = [], []
    for index in range(truth['order']):
        target_R = G @ rotation_y(index * 2 * np.pi / truth['order'])
        relative = target_R.T @ R
        angles.append(float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1)))))
        adds.append(float(np.linalg.norm(xyz @ R.T+t - (xyz @ target_R.T+target_t), axis=1).mean())*100)
    translation = float(np.linalg.norm(t-target_t)*100)
    rotation = min(angles)
    return dict(translation_cm=translation, rotation_deg=rotation, ADDsym_cm=min(adds),
                success_5cm_5deg=translation < 5 and rotation < 5)


def distribution(values):
    x = np.asarray(values, float)
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                P90=float(np.quantile(x, .9)) if len(x) else None,
                std=float(x.std(ddof=1)) if len(x)>1 else None,
                max=float(x.max()) if len(x) else None)


def paired(rows, reference, repeats=5000):
    delta = {row['id']: row['metric']['translation_cm']-reference[row['id']]['metric']['translation_cm'] for row in rows}
    sessions = sorted({row['session'] for row in rows})
    by_session = {s:[row for row in rows if row['session']==s] for s in sessions}
    rng = np.random.default_rng(20261013)
    draws = rng.integers(0,len(sessions),size=(repeats,len(sessions)))
    summary = {}
    for metric in ('translation_cm','rotation_deg','ADDsym_cm','success_5cm_5deg'):
        chunks = [np.asarray([float(row['metric'][metric])-float(reference[row['id']]['metric'][metric])
                            for row in by_session[session]]) for session in sessions]
        sums = np.asarray([x.sum() for x in chunks]); counts = np.asarray([len(x) for x in chunks])
        boot = sums[draws].sum(1) / counts[draws].sum(1)
        x = np.concatenate(chunks)
        summary[metric] = dict(mean_difference=float(x.mean()),
                               CI95=np.quantile(boot,[.025,.975]).tolist(),
                               improved=int((x>0).sum()) if metric.startswith('success') else int((x<0).sum()),
                               worsened=int((x<0).sum()) if metric.startswith('success') else int((x>0).sum()))
    return dict(frames=len(rows), sessions=len(sessions), repeats=repeats, by_metric=summary)


def appearance_metrics(rows, metadata):
    matrix=Counter(); records=[]
    for row in rows:
        truth = metadata[row['id']]['human_states_native']
        for index, (actual, state) in enumerate(zip(truth,row['appearance']['states'])):
            matrix[(actual,state)] += 1
            if actual in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED'):
                records.append(dict(id=row['id'],corner=index,actual=actual,state=state,
                    score=row['appearance']['visibility_score'][index]))
    actual=np.asarray([r['actual']=='DIRECT_VISIBLE' for r in records])
    predicted=np.asarray([r['state'] in ('VISIBLE_BOUNDARY','VISIBLE_INTERIOR') for r in records])
    score=np.asarray([r['score'] for r in records])
    tp=int((actual&predicted).sum());fp=int((~actual&predicted).sum());fn=int((actual&~predicted).sum());tn=int((~actual&~predicted).sum())
    positive=score[actual];negative=score[~actual]
    auc=float(((positive[:,None]>negative[None,:])+0.5*(positive[:,None]==negative[None,:])).mean()) if len(positive) and len(negative) else None
    return dict(state_counts=[dict(actual=a,predicted=b,count=n) for (a,b),n in sorted(matrix.items())],
        direct_vs_external=dict(TP=tp,FP=fp,FN=fn,TN=tn,visible_precision=tp/max(1,tp+fp),
            visible_recall=tp/max(1,tp+fn),external_recall=tn/max(1,tn+fp),visibility_AUC=auc),
        known_direct_visible=distribution(positive),known_external_occluded=distribution(negative),
        limitations='Fixed historical N3 correspondence phase; coordinate errors can cause false absence. Boundary/interior has no human ground truth here. Color cannot establish depth ownership.',
        rows=records)


def score(output):
    seal=C.read(output/'PREDICTION_SEAL.json')
    assert seal['before_scoring'] and seal['sha256']==C.sha(output/'PREDICTIONS.jsonl.gz')
    # First annotation access occurs only after prediction seal verification.
    targets=C.read(C.ROOT/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json')
    annotation=C.read(C.DOC/'SCORING_METADATA.json')
    metadata={row['id']:row for row in annotation['frames']}
    raw=list(C.rows(output/'PREDICTIONS.jsonl.gz'))
    values=[]; stages=[]
    for row in raw:
        row['metric']=pose_metric(row['actual_pose'],targets[row['id']]['truth'])
        values.append(row)
        if row['method'] in ('RGB_CURRICULUM','RGB_BOUNDARY_CURRICULUM','SEGMENTED_RGB_CURRICULUM','SEGMENTED_RGB_BOUNDARY_CURRICULUM'):
            initial=np.asarray(row['actual_pose']['R_cf'])
            Q=initial.T@np.asarray(row['actual_pose']['R_physical'])
            for stage in row['solution'].get('stages',[]):
                p=dict(row['actual_pose'],R_cf=stage['R'],centroid=stage['t'],R_physical=(np.asarray(stage['R'])@Q).tolist())
                stages.append(dict(id=row['id'],session=row['session'],arm=row['arm'],method=row['method'],stage=stage['name'],
                    accepted=stage['accepted'],used_ids=stage['used_ids'],reason=stage['reason'],metric=pose_metric(p,targets[row['id']]['truth'])))
    groups=defaultdict(list)
    for row in values:
        groups[(row['arm'],row['method'])].append(row)
    summary={}; comparisons={}; state={}
    for (arm,name), group in groups.items():
        summary[arm+'/'+name]=dict(frames=len(group),sessions=len({r['session'] for r in group}),
            accepted=sum(r['solution']['accepted'] for r in group),
            reasons=dict(Counter(r['solution']['reason'] for r in group)),
            metrics={metric:distribution([r['metric'][metric] for r in group]) for metric in ('translation_cm','rotation_deg','ADDsym_cm')},
            success_5cm_5deg=float(np.mean([r['metric']['success_5cm_5deg'] for r in group])))
        for reference_name in ('INITIAL','ALL_STANDARD','SELF_STANDARD','RGB_ONE_STAGE','SEGMENTED_RGB_ONE_STAGE','SEGMENTED_RGB_CURRICULUM'):
            if (arm,reference_name) not in groups:
                continue
            reference={r['id']:r for r in groups[(arm,reference_name)]}
            if name!=reference_name:
                comparisons[arm+'/'+name+'-'+reference_name]=paired(group,reference)
        if name in ('RGB_CURRICULUM','SEGMENTED_RGB_CURRICULUM'):
            state[arm+'/'+name]=appearance_metrics(group,metadata)
    C.save_rows(output/'SCORED_ROWS.jsonl.gz',values)
    C.save_rows(output/'STAGE_SCORES.jsonl.gz',stages)
    C.write(output/'RESULTS.json',dict(partition=C.read(output/'PROTOCOL.json')['partition'],summary=summary,
        paired_comparisons=comparisons,appearance=state,
        limitations='Historical DEV revisited, session split protects only this new tuning. Pose reference reconstructed from existing annotation, not independently measured. Cached detector/N3 inference; timing is new postprocessing only.',
        sealed_predictions_sha256=seal['sha256']))
    for key, item in summary.items():
        print(key,'accepted',item['accepted'],'T',round(item['metrics']['translation_cm']['mean'],4),
              'R',round(item['metrics']['rotation_deg']['mean'],4),'success',round(item['success_5cm_5deg'],4),flush=True)
    for arm, item in state.items():
        print('STATE',arm,item['direct_vs_external'],flush=True)
