"""Prediction-only global-mask comparison of the sealed prior/no-prior poses.

All95 choices are recorded before reused annotation metrics are read. This
does not optimize poses or alter keypoint/state/model/region observations.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'

from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_visibility_pose_v4_8 as P
from .global_pallet_silhouette_v4_9 import observe_mask_outline, choose_by_outline
from .scoring import pose_metric, distribution


OUTPUT=P.OUTPUT.parent/'outline_candidate_audit_v4_9'
MASKS=P.OUTPUT.parent/'image_evidence_v4_6'


def main():
    if OUTPUT.exists():raise FileExistsError('Preserve previous audit')
    protocol=C.read(P.OUTPUT/'PROTOCOL.json')
    E._verify_sources(protocol['source_sha256']);E._verify_sources(protocol['input_sha256'])
    E._verify_sources(protocol['protected_user_sha256'])
    maskseal=C.read(MASKS/'SEAL.json')
    assert C.sha(MASKS/'RECORDS.json')==maskseal['records_sha256']
    assert C.sha(MASKS/'PROTOCOL.json')==maskseal['protocol_sha256']
    meta={r['id']:r for r in C.read(MASKS/'RECORDS.json')}
    rows=[];bindings={}
    for part,count in (('CAL',46),('TEST',49)):
        journal=P.OUTPUT/f'{part}_PREDICTIONS.jsonl.gz';sealpath=P.OUTPUT/f'{part}_PREDICTION_SEAL.json'
        seal=C.read(sealpath)
        assert C.sha(journal)==seal['predictions_sha256']
        assert C.sha(P.OUTPUT/'PROTOCOL.json')==seal['protocol_sha256']
        values=list(C.rows(journal));assert len(values)==count
        rows+=values
        for path in (journal,sealpath):bindings[str(path.relative_to(C.ROOT))]=C.sha(path)
    assert len(rows)==95 and set(meta)=={r['id'] for r in rows}
    for path in (MASKS/'SEAL.json',MASKS/'RECORDS.json',MASKS/'PROTOCOL.json',P.OUTPUT/'PROTOCOL.json'):
        bindings[str(path.relative_to(C.ROOT))]=C.sha(path)
    sources=deepcopy(protocol['source_sha256'])
    for name in (Path(__file__).name,'global_pallet_silhouette_v4_9.py'):
        path=Path(__file__).parent/name;sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    OUTPUT.mkdir(parents=True)
    C.write(OUTPUT/'PROTOCOL.json',dict(schema='GLOBAL_MASK_PRIOR_NO_PRIOR_CANDIDATE_AUDIT_V4_9',
        ids=[r['id'] for r in rows],CAL=46,TEST=49,severe=0,
        source_sha256=sources,input_sha256=bindings,protected_user_sha256=protocol['protected_user_sha256'],
        rule='supported joint-feasible candidates only; observed-hull IoU + symmetric median distance/diagonal',
        source_mask='frozen full-object-box SAM v4.6 clean derivative',
        objective_is_calibrated_probability=False,GT_or_record_ids_in_selection=False,
        S_or_U_local_queries=0,new_pose_fits=0,new_mask_forwards=0,
        annotation_metrics_only_after_all95_choice_seal=True,TEST_is_reused_exploratory=True))
    values=[];started=time.monotonic()
    for index,row in enumerate(rows):
        record=meta[row['id']];maskpath=MASKS/record['mask_npz']
        assert C.sha(maskpath)==record['mask_file_sha256'] and record['image_sha256']==row['image_sha256']
        with np.load(maskpath) as masks:mask=masks['pallet_mask_clean'].copy()
        g=row['runtime']['initial_geometry'];obs=observe_mask_outline(mask,g['image_hw'])
        candidates=[]
        for arm in P.ARMS:
            result=row['arms'][arm]
            if result['accepted']:
                pose=result['actual_pose']
                candidates.append(dict(name=arm,xyz=g['native_xyz'],K=g['K'],R=pose['R_cf'],t=pose['centroid']))
        choice=choose_by_outline(obs,candidates)
        selected=choice['selected_name'] or P.ARMS[0]
        values.append(dict(id=row['id'],partition=row['partition'],image=row['image'],states=row['states'],
            selected_arm=selected,outline_selection=choice,
            selected_result=row['arms'][selected],S_queries=0,U_queries=0,GT_inputs=False,
            mask_file_sha256=record['mask_file_sha256']))
        print(f"{index+1}/95 {row['id']} selected={selected}",flush=True)
    C.save_rows(OUTPUT/'PREDICTIONS.jsonl.gz',values)
    E._verify_sources(sources);E._verify_sources(bindings);E._verify_sources(protocol['protected_user_sha256'])
    C.write(OUTPUT/'PREDICTION_SEAL.json',dict(rows=95,predictions_sha256=C.sha(OUTPUT/'PREDICTIONS.jsonl.gz'),
        protocol_sha256=C.sha(OUTPUT/'PROTOCOL.json'),before_new_annotation_metrics=True,
        source_sha256=sources,input_sha256=bindings,elapsed_seconds=time.monotonic()-started))
    reports={}
    for part in ('CAL','TEST'):
        target=E.COHORT/('OOF_pose_v1/ISOLATED_CAL_SCORING.json' if part=='CAL' else 'test_reused_v1/ISOLATED_TEST_SCORING.json')
        labels=C.read(target)['rows'];scored=[]
        for value in values:
            if value['partition']!=part:continue
            scored.append(dict(value,metric=pose_metric(value['selected_result']['actual_pose'],labels[value['id']]['target']['truth'])))
        report=dict(partition=part,frames=len(scored),translation_cm=distribution([r['metric']['translation_cm'] for r in scored]),
            rotation_deg=distribution([r['metric']['rotation_deg'] for r in scored]),
            translation_le5=sum(r['metric']['translation_cm']<=5 for r in scored),
            successes_5cm_5deg=sum(r['metric']['success_5cm_5deg'] for r in scored),
            fallback_count=sum(not r['selected_result']['accepted'] for r in scored),rows=scored)
        C.write(OUTPUT/f'{part}_SCORE.json',report);reports[part]=report
    print({part:{k:v for k,v in report.items() if k!='rows'} for part,report in reports.items()},flush=True)


if __name__=='__main__':main()
