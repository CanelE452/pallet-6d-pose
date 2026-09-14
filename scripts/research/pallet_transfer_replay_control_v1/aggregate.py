"""Frozen primary contrasts and descriptive panel aggregation."""
import json
from pathlib import Path
import numpy as np

from runtime import ROOT,RAW,DOC,atomic_json
from evaluate import NAMES

ARMS=('T8_FULL','T8_QUARTER','REPLAY','T32_COMPUTE')
CONTRASTS={'C_main':('REPLAY','T8_QUARTER'),'C_pract':('REPLAY','T8_FULL'),
           'C_budget':('REPLAY','T32_COMPUTE'),'C_scale':('T8_QUARTER','T8_FULL')}
BOOT=10000;SEED=20260914


def read(path):return json.loads(Path(path).read_text())


def pck(rows,weights=None):
    values=list(rows.values()); w=np.ones(len(values)) if weights is None else np.asarray(weights,float)
    numerator=sum(weight*row['numerators']['10.0'] for weight,row in zip(w,values))
    denominator=sum(weight*row['denominator'] for weight,row in zip(w,values))
    return float(numerator/denominator) if denominator else float('nan')


def bootstrap(rowsets,clusters,scheme):
    """One draw is shared across all 13 models and all contrasts."""
    keys=sorted(next(iter(rowsets.values())))
    assert all(sorted(rows)==keys for rows in rowsets.values())
    groups=sorted(set(clusters[k] for k in keys));index={g:i for i,g in enumerate(groups)}
    group_index=np.asarray([index[clusters[k]] for k in keys]);rng=np.random.default_rng(SEED)
    out={name:[] for name in CONTRASTS}; arm_samples={arm:[] for arm in ARMS};versus_r0={arm:[] for arm in ARMS};r0=[]
    for _ in range(BOOT):
        counts=rng.multinomial(len(groups),np.full(len(groups),1/len(groups))) if scheme=='cluster' else rng.multinomial(len(keys),np.full(len(keys),1/len(keys)))
        weights=counts[group_index] if scheme=='cluster' else counts
        base=pck({k:rowsets['R0'][k] for k in keys},weights);r0.append(base)
        arm_values={}
        for arm in ARMS:
            value=float(np.mean([pck({k:rowsets[f'{arm}_seed{s}'][k] for k in keys},weights) for s in (1,2,3)]))
            arm_values[arm]=value;arm_samples[arm].append(value);versus_r0[arm].append(value-base)
        for label,(left,right) in CONTRASTS.items():out[label].append(arm_values[left]-arm_values[right])
    return dict(scheme=scheme,units=len(groups) if scheme=='cluster' else len(keys),resamples=BOOT,seed=SEED,
        contrasts={label:dict(low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)),
            fraction_positive=float(np.mean(np.asarray(values)>0)),excludes_zero=bool(np.quantile(values,.025)>0 or np.quantile(values,.975)<0)) for label,values in out.items()},
        R0=dict(low=float(np.quantile(r0,.025)),high=float(np.quantile(r0,.975))),
        arms={arm:dict(low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975))) for arm,v in arm_samples.items()},
        versus_R0={arm:dict(low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975)),
            fraction_positive=float(np.mean(np.asarray(v)>0)),excludes_zero=bool(np.quantile(v,.025)>0 or np.quantile(v,.975)<0))
            for arm,v in versus_r0.items()})


def geometry(rows,keys):
    errors=np.concatenate([np.asarray(rows[k]['errors_px'],float) for k in keys]);means=[np.mean(rows[k]['errors_px']) for k in keys]
    return dict(frames=len(keys),points=len(errors),median_px=float(np.median(errors)),p90_px=float(np.quantile(errors,.9)),
        frame_mean_px=float(np.mean(means)),gross20=float(np.mean(errors>20)))


def summarize_result(value):
    target=value['target'];pose=value['pose'];source=value['source']
    return dict(target_ALL_GT_PCK5=target['ALL_GT_PCK']['5.0'],target_ALL_GT_PCK10=target['ALL_GT_PCK']['10.0'],
        target_ALL_GT_PCK20=target['ALL_GT_PCK']['20.0'],target_match_rate=target['detection_match_rate'],
        target_AP50=target['canonical_2d']['box_ap50'],target_AP50_95=target['canonical_2d']['box_ap50_95'],
        negative_AUROC=target['negative']['auroc'],negative_FPR95=target['negative']['fpr95'],
        padding_only_top1=target['padding_only_top1'],target_individual_matched=target['individual_matched'],
        translation_median_cm=pose['ALL']['translation_median_cm'],yaw_median_deg=pose['ALL']['yaw_median_deg'],
        rotation_median_deg=pose['ALL']['rotation_median_deg'],iou3d_median=pose['ALL']['iou3d_median'],
        add_sym_auc=pose['ALL']['add_sym_auc'],pose_coverage=pose['coverage'],
        source_ALL_GT_PCK5=source['ALL_GT_PCK']['5.0'],source_ALL_GT_PCK10=source['ALL_GT_PCK']['10.0'],
        source_ALL_GT_PCK20=source['ALL_GT_PCK']['20.0'],source_match_rate=source['detection_match_rate'],
        source_AP50=source['box_ap50'],source_AP50_95=source['box_ap50_95'],source_individual_matched=source['individual_matched'])


def main():
    results={name:read(RAW/'evaluation'/name/'RESULT.json') for name in NAMES}
    target_rows={name:read(RAW/'evaluation'/name/'TARGET_PER_FRAME.json') for name in NAMES}
    source_rows={name:read(RAW/'source_evaluation'/name/'SOURCE_PER_FRAME.json') for name in NAMES}
    target_keys=sorted(k for k,v in target_rows['R0'].items() if v['kind']=='positive')
    target_rows={n:{k:r[k] for k in target_keys} for n,r in target_rows.items()}
    source_keys=sorted(source_rows['R0'])
    target_common=[k for k in target_keys if all(target_rows[n][k]['matched'] and target_rows[n][k]['errors_px'] for n in NAMES)]
    source_common=[k for k in source_keys if all(source_rows[n][k]['matched'] and source_rows[n][k]['errors_px'] for n in NAMES)]
    per_seed={name:summarize_result(value) for name,value in results.items()}
    target_common_geometry={n:geometry(target_rows[n],target_common) for n in NAMES}
    source_common_geometry={n:geometry(source_rows[n],source_common) for n in NAMES}
    seed_means={}
    for arm in ARMS:
        rr=[per_seed[f'{arm}_seed{s}'] for s in (1,2,3)];keys=[k for k,v in rr[0].items() if isinstance(v,(float,int))]
        seed_means[arm]={k:float(np.mean([r[k] for r in rr])) for k in keys}
        for nested in ('target_individual_matched','source_individual_matched'):
            seed_means[arm][nested]={k:float(np.mean([r[nested][k] for r in rr])) for k in rr[0][nested]}
        seed_means[arm]['target_common_geometry']={k:float(np.mean([target_common_geometry[f'{arm}_seed{s}'][k] for s in (1,2,3)])) for k in ('median_px','p90_px','frame_mean_px','gross20')}
        seed_means[arm]['source_common_geometry']={k:float(np.mean([source_common_geometry[f'{arm}_seed{s}'][k] for s in (1,2,3)])) for k in ('median_px','p90_px','frame_mean_px','gross20')}
    point={label:dict(target_pck10=seed_means[left]['target_ALL_GT_PCK10']-seed_means[right]['target_ALL_GT_PCK10'],
        source_pck10=seed_means[left]['source_ALL_GT_PCK10']-seed_means[right]['source_ALL_GT_PCK10']) for label,(left,right) in CONTRASTS.items()}
    point['versus_R0']={arm:dict(target_pck10=seed_means[arm]['target_ALL_GT_PCK10']-per_seed['R0']['target_ALL_GT_PCK10'],
        source_pck10=seed_means[arm]['source_ALL_GT_PCK10']-per_seed['R0']['source_ALL_GT_PCK10']) for arm in ARMS}
    target_clusters={k:target_rows['R0'][k]['session'] for k in target_keys};source_clusters={k:source_rows['R0'][k]['scenario'] for k in source_keys}
    target_boot_cluster=bootstrap(target_rows,target_clusters,'cluster');target_boot_frame=bootstrap(target_rows,target_clusters,'frame')
    source_boot_cluster=bootstrap(source_rows,source_clusters,'cluster');source_boot_frame=bootstrap(source_rows,source_clusters,'frame')
    sessions=sorted(set(target_clusters.values()));loso={}
    for heldout in sessions:
        keys=[k for k in target_keys if target_clusters[k]!=heldout]
        arms={arm:float(np.mean([pck({k:target_rows[f'{arm}_seed{s}'][k] for k in keys}) for s in (1,2,3)])) for arm in ARMS}
        loso[heldout]=dict(R0=pck({k:target_rows['R0'][k] for k in keys}),arms=arms,
            contrasts={label:arms[left]-arms[right] for label,(left,right) in CONTRASTS.items()})
    atomic_json(DOC/'METRICS_PER_SEED.json',dict(status='COMPLETE',per_model=per_seed,R0=per_seed['R0'],seed_means=seed_means,
        target_common_frames=len(target_common),target_common_geometry=target_common_geometry,
        source_common_frames=len(source_common),source_common_geometry=source_common_geometry))
    atomic_json(DOC/'PAIRED_CONTRASTS.json',dict(status='COMPLETE',unit='absolute proportion; multiply by100 for percentage points',
        point_estimates=point,target=dict(session_cluster=target_boot_cluster,frame=target_boot_frame,leave_one_session_out=loso),
        source=dict(scenario_cluster=source_boot_cluster,frame=source_boot_frame),
        estimand='mean of three seed-specific arm PCK10 values on each shared resample; single R0 is not triplicated as independent',
        caveat='Historical DEV and marginal percentile intervals; resampling does not create new sessions or familywise coverage.'))
    print(json.dumps(dict(point_estimates=point,target_session=target_boot_cluster['contrasts'],source_scenario=source_boot_cluster['contrasts']),indent=2),flush=True)


if __name__=='__main__':main()
