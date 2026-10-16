"""Bound saved-row comparisons against N3 and actual hidden replacements.

Leaves immutable main metrics/protocol unchanged; no models, fits or truth IO.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np

NAME='pallet_kp_corrected_supervision_20261010_v1'
def rows(p):
    with gzip.open(p,'rt')as f:return [json.loads(l)for l in f]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p,root):return dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size)
def distribution(a):
    a=np.asarray(a,float);assert np.isfinite(a).all()
    return dict(n=len(a),mean=float(a.mean())if len(a)else None,sample_variance=float(a.var(ddof=1))if len(a)>1 else None,
        sample_std=float(a.std(ddof=1))if len(a)>1 else None,median=float(np.median(a))if len(a)else None,
        P90=float(np.quantile(a,.9))if len(a)else None,max=float(a.max())if len(a)else None,unit='px',ddof=1)
def error(q,g,valid):
    q=np.asarray(q,float)[:8];g=np.asarray(g,float)[:8]
    e=np.linalg.norm(q-g,axis=1);known=np.isfinite(q).all(1)&~(q==-1).all(1)&np.isfinite(g).all(1)
    known&=np.isin(np.arange(8),valid);e[~known]=np.nan
    return e,known
def quality(t,r):
    if t is None or r is None:return 'no_common_pose'
    if t < -1e-9 and r < -1e-9:return 'both_improved'
    if t > 1e-9 and r > 1e-9:return 'both_worsened'
    return 'mixed_or_equal'
def write(p,j):
    assert not p.exists(),'Preserve completed supplementary statistics'
    p.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);a=p.parse_args()
    root=a.root.resolve();doc=root/'_docs/experiments'/NAME
    cohort=json.loads((doc/'COHORT.json').read_text());ids=cohort['ids'];allowed=set(ids);labels={f['id']:f['label']for f in cohort['frames']}
    raw=rows(doc/'LEARNED_PREDICTIONS.jsonl.gz');post=rows(doc/'POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz')
    n3={r['id']:r for r in rows(doc/'HISTORICAL_FILTERED_ROWS.jsonl.gz')if r['alias']=='N3_SUBPIX'}
    lookup={(r['method'],r['id']):r for r in post};assert set(n3)==allowed
    records=[]
    for r in raw:
        fid=r['id'];ar=lookup[('CORRECTED_'+r['method'],fid)];comparator=n3[fid];pa,pb=r['pose'],comparator['pose']
        t=pa['translation_cm']-pb['translation_cm']if pa['available']and pb['available']else None
        rotation=pa['rotation_deg']-pb['rotation_deg']if pa['available']and pb['available']else None
        records.append(dict(id=fid,method='CORRECTED_'+r['method'],label=labels[fid],mask_applied=ar['mask_applied'],
            mask_wrong_on_known=ar['mask_wrong_on_known'],raw_status=r['output_status'],new_pose=r['new_pose_estimated'],
            fallback=r['fallback_used'],no_pose=r['no_pose'],translation_delta_cm=t,rotation_delta_deg=rotation,
            operational_pose_quality_vs_N3=quality(t,rotation)))
    def groups(rr):
        def s(rr):return dict(frames=len(rr),new_pose=sum(r['new_pose']for r in rr),fallback=sum(r['fallback']for r in rr),
            no_pose=sum(r['no_pose']for r in rr),quality_operational=dict(Counter(r['operational_pose_quality_vs_N3']for r in rr)),
            quality_new_only=dict(Counter(r['operational_pose_quality_vs_N3']for r in rr if r['new_pose'])),
            quality_fallback_only=dict(Counter(r['operational_pose_quality_vs_N3']for r in rr if r['fallback'])))
        return dict(all=s(rr),wrong_mask=s([r for r in rr if r['mask_wrong_on_known']]),
                    matching_known_mask=s([r for r in rr if not r['mask_wrong_on_known']]))
    scopes={'combined':ids,'easy':[i for i in ids if labels[i]=='clean'],'medium':[i for i in ids if labels[i]=='moderate']}
    mask=dict(schema='same_predicted_mask_outcomes_vs_fixed_N3_v1',complete=True,
        comparator='N3_SUBPIX',mask_predicate='Unchanged BASE-phase humanSELF disagreement; mask differences do not fail frames.',
        status_and_pose_quality_separate=True,rows=records,
        strata={s:{m:groups([r for r in records if r['id']in set(ii)and r['method']==m])for m in sorted({r['method']for r in records if r['mask_applied']})}for s,ii in scopes.items()},
        bindings=[bind(doc/n,root)for n in ('COHORT.json','METRICS.json','LEARNED_PREDICTIONS.jsonl.gz','POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz','HISTORICAL_FILTERED_ROWS.jsonl.gz')],
        new_detector_head_PnP_ray_calls=0)
    write(doc/'MASK_OUTCOME_N3.json',mask)
    damage={};hidden={};arms=sorted({r['method']for r in records})
    for scope,ii in scopes.items():
        scope_set=set(ii);damage[scope]={};hidden[scope]={}
        for m in arms:
            rr=[r for r in post if r['id']in scope_set and r['method']==m]
            pairs={};frame_pairs={};counts=Counter();replacement=[]
            for ar in rr:
                before,known=error(n3[ar['id']]['native_points'],ar['reference_native_points_px'],ar['reference_valid_native_ids'])
                after,after_known=error(ar['output_native_points'],ar['reference_native_points_px'],ar['reference_valid_native_ids'])
                assert not ar['hidden_reprojected_ids']or ar['new_pose_estimated']
                n3phase=lookup[('N3_SUBPIX',ar['id'])]['permutation_native_to_canonical']
                counts['frames_same_BASE_and_N3_reference_phase']+=n3phase==ar['permutation_native_to_canonical']
                local={}
                for k in range(8):
                    category=ar['human_states_native'][k]
                    if known[k]and after_known[k]:
                        pairs.setdefault(category,[]).append((before[k],after[k]));local.setdefault(category,[]).append((before[k],after[k]))
                    if category!='SELF_OCCLUDED':continue
                    counts['human_SELF_ids']+=1
                    if not ar['new_pose_estimated']:counts['SELF_without_new_pose']+=1;continue
                    if k not in ar['hidden_reprojected_ids']:counts['SELF_not_reprojected_despite_new_pose']+=1;continue
                    counts['SELF_reprojected_after_new_pose']+=1
                    base,base_known=error(ar['frozen_BASE_native_points'],ar['reference_native_points_px'],ar['reference_valid_native_ids'])
                    if not base_known[k]or not known[k]or not after_known[k]:counts['SELF_reprojected_missing_valid_reference_or_coordinates']+=1;continue
                    replacement.append((float(base[k]),float(before[k]),float(after[k])))
                for category,values in local.items():frame_pairs.setdefault(category,[]).append(np.mean(values,axis=0))
            damage[scope][m]={}
            for category,values in pairs.items():
                before,after=np.asarray(values).T;fb,fa=np.asarray(frame_pairs[category]).T
                damage[scope][m][category]=dict(corners=len(values),frames=len(frame_pairs[category]),before_N3=distribution(before),
                    after=distribution(after),paired_delta_vs_N3=distribution(after-before),before_N3_frame_mean=distribution(fb),after_frame_mean=distribution(fa),
                    improved=int((after<before-1e-9).sum()),worsened=int((after>before+1e-9).sum()),
                    good5_to_bad10=int(((before<5)&(after>10)).sum()),bad20_to_good10=int(((before>20)&(after<=10)).sum()))
            if replacement:
                base,before,after=np.asarray(replacement).T
            else:base=before=after=np.array([])
            hidden[scope][m]=dict(counts=dict(counts),evaluable_reprojected_SELF_corners=len(replacement),before_BASE=distribution(base),
                before_N3_same_BASE_phase=distribution(before),after_reprojection=distribution(after),
                delta_vs_BASE=distribution(after-base),delta_vs_N3=distribution(after-before))
    write(doc/'CORNER_N3_AND_REPROJECTED_SELF.json',dict(schema='frozen_BASE_phase_corner_damage_vs_N3_v1',complete=True,
        reference='Same fixed BASE phase for both N3 input coordinates and corrected output; no best-branch reselection.',
        visibility_damage_vs_N3=damage,actual_reprojected_SELF_only=hidden,
        exclusions='ActualSELF replacement requires NEW_POSE and ID in reprojected_ids; missing references and no-new outputs counted separately.',
        bindings=mask['bindings'],new_models_fits_rays_truth_reads=0))
    print(json.dumps(dict(complete=True,mask_groups=mask['strata']['combined']['CORRECTED_IMAGE_ROLE'])))

if __name__=='__main__':main()
