"""Predeclared student CVaR/paired-tail gates; no metric selection."""
import math
import numpy as np
from contracts import *
from risk import cvar90
from stats import interval
from mechanism import summary

CONTROLS=('random','diversity','geometry_weighted_diversity')
METHODS=(*CONTROLS,'proposed')


def geometry(rows,ids):
    values=np.concatenate([rows[k]['errors_px'] for k in sorted(ids)])
    return dict(frames=len(ids),corners=len(values),median_px=float(np.median(values)),
        p90_px=float(np.quantile(values,.9)),gross20=float(np.mean(values>20)))


def tail(values):
    if any(v is None or not np.isfinite(v) for v in values):return None
    assert len(values)==145
    result=cvar90(values)
    assert abs(result-float(np.mean(sorted(values,reverse=True)[:15])))<1e-10, 'Independent fixed145/worst15 parity'
    return result


def bootstrap(poses,split):
    groups=np.array([r['capture_session'] for r in split['evaluation']]);unique=np.unique(groups)
    groups_idx=[np.flatnonzero(groups==g) for g in unique];rng=np.random.default_rng(SEED)
    by={n:{k:np.array([np.nan if r[k] is None else r[k] for r in rows],float) for k in ('translation_cm','yaw_deg')} for n,rows in poses.items()}
    draws={c:{k:{str(s):[] for s in (1,2,3)} for k in ('translation_cm','yaw_deg')} for c in ('diversity','geometry_weighted_diversity')}
    mean_draws={c:{k:[] for k in ('translation_cm','yaw_deg')} for c in draws}
    for _ in range(BOOTSTRAPS):
        ii=np.concatenate([groups_idx[j] for j in rng.integers(len(unique),size=len(unique))])
        for c in draws:
            for k in draws[c]:
                deltas=[]
                for seed in (1,2,3):
                    a,b=by[f'proposed_seed{seed}'][k][ii],by[f'{c}_seed{seed}'][k][ii]
                    delta=cvar90(a)-cvar90(b) if np.isfinite(a).all() and np.isfinite(b).all() else None
                    draws[c][k][str(seed)].append(delta);deltas.append(delta)
                mean_draws[c][k].append(float(np.mean(deltas)) if all(d is not None for d in deltas) else None)
    return dict(draws=BOOTSTRAPS,seed=SEED,session_count=len(unique),sessions=unique.tolist(),
        resampling='Paired sessions; all frames of sampled sessions; recompute worst ceil(.1*N) every draw; same draw across seeds; seeds are not independent sessions',
        per_seed={c:{k:{s:interval(v) for s,v in vv.items()} for k,vv in d.items()} for c,d in draws.items()},
        seed_mean={c:{k:interval(v) for k,v in d.items()} for c,d in mean_draws.items()})


def main():
    verify_lock();assert read(DOC/'TRAINING_AUDIT.json')['fits']==4
    poses=read(RAW/'PER_FRAME_POSE.json');split=read(DOC/'SPLIT_BINDING.json')
    results={};rows={};metrics={};new=read(DOC/'TRAINING_AUDIT.json')['audits']
    for name,pose_rows in poses.items():
        folder=(RAW if name in new else OLD_RAW)/'evaluation'/name
        results[name]=read(folder/'RESULT.json');rows[name]=read(folder/'PER_FRAME.json')
        assert len(rows[name])==len(pose_rows)==145
        assert [r['frame_id'] for r in pose_rows]==[r['frame_id'] for r in split['evaluation']]
        assert results[name]['actual_evaluation_positive']==145 and results[name]['negative_count']==2689
        metrics[name]=dict(translation_CVaR90_cm=tail([r['translation_cm'] for r in pose_rows]),
            yaw_CVaR90_deg=tail([r['yaw_deg'] for r in pose_rows]),
            kp_frame_mean_CVaR90_px=tail([float(np.mean(r['errors_px'])) if r['errors_px'] else None for r in rows[name].values()]),
            translation_median_cm=results[name]['pose']['ALL']['translation_median_cm'],
            yaw_median_deg=results[name]['pose']['ALL']['yaw_median_deg'],
            pose_coverage=results[name]['pose']['coverage'],Det=results[name]['Det'],
            AP50_95=results[name]['metrics']['box_ap50_95'],AUROC=results[name]['auroc'],FPR95=results[name]['fpr95'],
            IoU3D=results[name]['pose']['ALL']['iou3d_median'],ADDsym_AUC=results[name]['pose']['ALL']['add_sym_auc'],
            individual_matched_geometry=results[name]['geometry'])
    paired=[]
    for seed in (1,2,3):
        names=[f'{m}_seed{seed}' for m in METHODS]
        common=set.intersection(*[{k for k,r in rows[n].items() if r['matched'] and r['errors_px']} for n in names])
        gm={m:geometry(rows[f'{m}_seed{seed}'],common) for m in METHODS}
        for m,g in gm.items():metrics[f'{m}_seed{seed}']['mandatory_four_arm_common_geometry']=g
        paired.append(dict(seed=seed,common_frames=len(common),common_ids=sorted(common),geometry=gm))
    means={}
    keys=[k for k,v in metrics['proposed_seed1'].items() if not isinstance(v,dict)]
    for method in METHODS:
        rr=[metrics[f'{method}_seed{s}'] for s in (1,2,3)]
        means[method]={k:float(np.mean([r[k] for r in rr])) if all(r[k] is not None for r in rr) else None for k in keys}
        means[method]['common_geometry']={k:float(np.mean([p['geometry'][method][k] for p in paired])) for k in ('median_px','p90_px','gross20')}
    proposed=means['proposed'];gates={};directions={}
    for tag,control in [('A','diversity'),('B','geometry_weighted_diversity')]:
        directions[control]=[metrics[f'proposed_seed{s}']['translation_CVaR90_cm'] is not None and metrics[f'{control}_seed{s}']['translation_CVaR90_cm'] is not None and metrics[f'proposed_seed{s}']['translation_CVaR90_cm']<metrics[f'{control}_seed{s}']['translation_CVaR90_cm'] for s in (1,2,3)]
        gates[tag]=proposed['translation_CVaR90_cm'] is not None and means[control]['translation_CVaR90_cm'] is not None and proposed['translation_CVaR90_cm']<means[control]['translation_CVaR90_cm'] and sum(directions[control])>=2
    gates['C']=all(means[m]['yaw_CVaR90_deg'] is not None for m in ('proposed','diversity','geometry_weighted_diversity')) and proposed['yaw_CVaR90_deg']<=max(means[m]['yaw_CVaR90_deg'] for m in ('diversity','geometry_weighted_diversity'))
    gates['D']=all(proposed['common_geometry']['p90_px']<=means[m]['common_geometry']['p90_px'] for m in ('random','diversity'))
    gates['E']=proposed['Det']>=max(means[m]['Det'] for m in CONTROLS)-1/145 and all(means[m]['pose_coverage']==1 for m in METHODS)
    gates['F']=True
    verdict='TASK_RISK_AL_DEVELOPMENT_SIGNAL' if all(gates.values()) else 'TASK_RISK_AL_NO_SIGNAL'
    write(DOC/'PER_SEED_RESULTS.json',dict(per_run=metrics,seed_means=means,paired_common_geometry=paired,
        old_instability_in_per_run_only=True,common_population='per-seed random/diversity/old_geometry/proposed intersection; primary pose CVaR uses all145, never this intersection'))
    write(DOC/'CVAR_RESULTS.json',dict(primary='translation_CVaR90_cm',definition='worst15 of fixed145',
        seed_means=means,per_seed_improvement=directions,gates=gates,scientific_verdict=verdict))
    write(DOC/'SESSION_BOOTSTRAP.json',bootstrap(poses,split))
    write(DOC/'FULL174_REFERENCE_RESULT.json',dict(name='FULL174_FIXED_COMPUTE_REFERENCE',seed=1,
        metrics=metrics['full174_seed1'],upper_bound=False,one_seed_descriptive=True,
        real_slots=2400,labels=174,slots_per_label=2400/174,selected30_slots_per_label=80))
    choices=read(DOC/'SELECTION_LOCK.json');write(DOC/'TASK_RISK_SELECTION.json',choices)
    pool_errors=read(RAW/'POOL_GT_ERRORS.json');risk=read(RAW/'TASK_RISK.json')
    selected=set(choices['selections']['proposed'])
    old=read(OLD_DOC/'SELECTION_LOCK.json')['selections'];old_a=set(old['diversity']);old_b=set(old['geometry_weighted_diversity'])
    write(DOC/'SELECTION_MECHANISM_LINK.json',dict(groups={name:summary([r for r in pool_errors if r['frame_id'] in ids]) for name,ids in [('proposed',selected),('diversity_unique',old_a-old_b),('geometry_unique',old_b-old_a)]},
        selected_risk_discrete_means={name:{k:float(np.mean([r[k] for r in risk if r['frame_id'] in ids])) for k in ('candidate_switch_rate','axis_switch_rate','pnp_failure_rate','R_task')} for name,ids in [('proposed',selected),('diversity_unique',old_a-old_b),('geometry_unique',old_b-old_a)]},
        causal_mediation_proven=False,explanation='Association between acquisition hardness and paired student tails, not causal mediation proof'))
    write(DOC/'VERDICT.json',dict(scientific_verdict=verdict,stage0='TASK_RISK_MECHANISM_PASS',
        gates=gates,new_student_fits=4,new_optimizer_updates=1200,old_verdict='RETROSPECTIVE_AL_NO_SIGNAL',
        old_verdict_unchanged=True,development_only=True,independent_confirmation=False,novelty_claim=False,
        paper_final_modified=False,automatic_additional_experiments=False))
    print(verdict,gates,flush=True)
    for m,r in means.items():print(m,r,flush=True)


if __name__=='__main__':main()
