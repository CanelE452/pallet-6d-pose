"""Postseal candidate availability/regret; no new selector or pose generation."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from scripts.research.pallet_risk_information_v1.common import read,rows,write,csv_write,sha
from scripts.research.pallet_accuracy_closeout_20261017_v1.baseline_audit import distribution
from .bank import read as bank_read
from .evaluate import project_reference
from .registry import SEEDS,POPULATIONS


def run(root,upstream,bank_root,reference_dir,output):
    root,output=Path(root),Path(output);seal=read(root/'PREDICTION_MANIFEST.json')
    if not seal['before_scoring'] or seal['real_reference_inputs']: raise ValueError('Postseal only')
    pred={r['id']:r for r in rows(root/'PREDICTIONS.jsonl.gz')};references=read(Path(reference_dir)/'FIXED_REFERENCES.json')['references']
    records=[];frames=[]
    for ordinal,frame in enumerate(rows(upstream)):
        fid=frame['id'];bank=bank_read(Path(bank_root)/'frames/real'/f'{ordinal:04d}'/'BANK.json.gz')
        aligned,phase=project_reference(frame,references[frame['partition']][fid])
        for seed in SEEDS:
            C=f'OBS_COMMON_s{seed}';B=f'OBS_BI_s{seed}';p=pred[fid]['predictions']
            cchoices={r['native_corner']:r for r in p[C]['choices']};bchoices={r['native_corner']:r for r in p[B]['choices']}
            cf=[];changed=0
            for corner in bank['corners']:
                k=corner['native_corner'];state=bank['states'][k]
                if state not in ('B','I'):continue
                cc,bc=cchoices[k],bchoices[k];partial=corner['partial_line'] is not None
                if partial:continue
                changed+=cc['applied_xy']!=bc['applied_xy']
                if aligned is None or not aligned[1][k]:continue
                errors=[float(np.linalg.norm(np.asarray(c['applied_xy'])-aligned[0][k])) for c in corner['candidates']]
                e0=errors[0];best=min(errors);ce=errors[cc['selected_candidate_index']];be=errors[bc['selected_candidate_index']]
                value=dict(id=fid,partition=frame['partition'],seed=seed,native_corner=k,automatic_state=state,
                    reference_kind='POSTSEAL_VIRTUAL_CUBOID_CANDIDATE_DIAGNOSTIC_NOT_PHYSICAL_EDGE_NOT_AUTOMATIC_POLICY',
                    candidate_count=len(errors),q0_error_px=e0,best_available_candidate_error_px=best,
                    has_strictly_better_than_q0=best<e0,COMMON_error_px=ce,BI_error_px=be,
                    COMMON_regret_px=ce-best,BI_regret_px=be-best,chosen_coordinate_changed=cc['applied_xy']!=bc['applied_xy'])
                records.append(value);cf.append(value)
            frames.append(dict(id=fid,partition=frame['partition'],seed=seed,full_corner_reference_n=len(cf),
                changed_coordinates=changed,COMMON_mean_corner_error_px=float(np.mean([v['COMMON_error_px'] for v in cf])) if cf else None,
                BI_mean_corner_error_px=float(np.mean([v['BI_error_px'] for v in cf])) if cf else None))
    summaries=[]
    for pop,parts in POPULATIONS.items():
        for seed in SEEDS:
            for state in ('B','I'):
                rr=[r for r in records if r['partition'] in parts and r['seed']==seed and r['automatic_state']==state]
                summaries.append(dict(population=pop,seed=seed,automatic_state=state,full_reference_corners=len(rr),
                    alternatives_n=sum(r['candidate_count']>1 for r in rr),good_alternative_n=sum(r['has_strictly_better_than_q0'] for r in rr),
                    q0=distribution([r['q0_error_px'] for r in rr],len(rr)),
                    best_candidate=distribution([r['best_available_candidate_error_px'] for r in rr],len(rr)),
                    COMMON_regret=distribution([r['COMMON_regret_px'] for r in rr],len(rr)),BI_regret=distribution([r['BI_regret_px'] for r in rr],len(rr)),
                    differing_selected_coordinates=sum(r['chosen_coordinate_changed'] for r in rr)))
    scored={r['id']:r for r in rows(root/'evaluation/SCORED_ROWS.jsonl.gz')};transfer=[]
    for pop,parts in POPULATIONS.items():
        for seed in SEEDS:
            ff=[r for r in frames if r['partition'] in parts and r['seed']==seed];C=f'OBS_COMMON_s{seed}';B=f'OBS_BI_s{seed}'
            local_better=[r for r in ff if r['BI_mean_corner_error_px'] is not None and r['BI_mean_corner_error_px']<r['COMMON_mean_corner_error_px']]
            transfer.append(dict(population=pop,seed=seed,frames=len(ff),frames_with_any_coordinate_change=sum(r['changed_coordinates']>0 for r in ff),
                local_reference_mean_better_n=len(local_better),
                local_better_but_position_worse_ids=[r['id'] for r in local_better if scored[r['id']]['metrics'][B]['translation_cm']>scored[r['id']]['metrics'][C]['translation_cm']],
                local_better_but_rotation_worse_ids=[r['id'] for r in local_better if scored[r['id']]['metrics'][B]['rotation_deg']>scored[r['id']]['metrics'][C]['rotation_deg']]))
    output.mkdir(parents=True,exist_ok=False);csv_write(output/'CANDIDATE_REFERENCE_DIAGNOSTIC.csv.gz',records)
    write(output/'CANDIDATE_DIAGNOSTIC_SUMMARY.json',dict(summaries=summaries,transfer=transfer,source_sha256=sha(__file__),
        automatic_outputs_unchanged=True,new_pose_solves=0,new_fits=0,reference_oracle_is_not_validated_new_selector=True))


def main():
    p=argparse.ArgumentParser()
    for name in ('run-dir','upstream','bank-root','reference-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.run_dir,a.upstream,a.bank_root,a.reference_dir,a.output)


if __name__=='__main__':main()
