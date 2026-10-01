"""Seal the bounded paired scorer experiment only after TRAIN feasibility."""
from pathlib import Path
from . import common as C


def main():
    gate=C.read(C.DOC/'SOURCE_TRAIN_GATE.json')
    assert gate['complete'] and gate['PASS'] is True
    assert gate['frames']==2598 and gate['VAL_quality_scored'] is False
    assert not (C.DOC/'TRAIN_PROTOCOL.json').exists()
    inputs=dict(features=C.bind(C.RAW/'SOURCE_FEATURES.npz'),
        train_labels=C.bind(C.RAW/'SOURCE_TRAIN_LABELS.npz'),
        train_gate=C.bind(C.DOC/'SOURCE_TRAIN_GATE.json'),
        feasibility_protocol=C.bind(C.DOC/'TRAIN_FEASIBILITY_PROTOCOL.json'),
        source_predictions_lock=C.bind(C.DOC/'SOURCE_PREDICTIONS_LOCK.json'),
        feature_lock=C.bind(C.DOC/'SOURCE_FEATURE_LOCK.json'),
        source_contract=C.bind(C.DOC/'SOURCE_CONTRACT.json'),
        runtime_amendment=C.bind(C.DOC/'SOURCE_RUNTIME_AMENDMENT_01.json'))
    codes=[C.bind(C.HERE/n) for n in ['common.py','source_features.py','train.py','source_val.py','seal_training.py']]
    p=dict(schema='pallet_pose_union_train_v1',created_at=C.now(),complete=True,
        arms=['R0_ONLY','UNION'],seeds=[1,2,3],train_rows=2598,
        feature_dim=94,epochs=30,batch_size=256,updates_per_fit=330,device='cpu',max_fits=6,
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,betas=[.9,.999],eps=1e-8),
        normalization=dict(source='R0_eligible_train_valid_candidates',std_floor=1e-6),
        inputs=inputs,codes=codes,
        candidate_pool='R0_ONLY: R0 long/short. UNION seed s: R0 long/short plus frozen DIVERSE251_s long/short. Entire R,t selected together.',
        architecture='Shared linear94 scalar cost; lower score wins. No expert ID or image features. All YOLO/PoseFix weights remain frozen.',
        target='Whole-pose min max(T/sT,R/sR) on eligible source TRAIN; exact target-cost ties Pareto then R0 then hypothesis name.',
        objective='Masked cross entropy of negative scalar scores; no-valid rows have target -1 and zero loss.',
        inference_tie='Exact learned score tie: R0 then hypothesis name, with no reference access.',
        no_valid_fallback='Both arms use the same saved R0 GEO whole pose when zero candidate is usable; if unavailable, mark failed and both errors +infinity.',
        paired_controls='Same seed initial parameters, normalized R0 features, and row batches across arms; all eligible2598 used each epoch. No extra seed or winner selection.',
        checkpoint='Final epoch30 only; no early stopping or VAL/real metric during fit.',
        source_val=dict(frames=1024,seeds=[1,2,3],median_strict=True,p90_ratio_max=1.05,
            failure_no_increase=True,comparators=['paired_R0_ONLY','R0_GEO','paired_DIVERSE_GEO'],
            all_seeds_required=True,no_checkpoint_selection=True,
            routing_lock_before_source_labels=True,real_route_requires_gate_pass=True,
            zero_valid_fallback='R0_GEO_if_available_else_failure'),
        real_evaluation=dict(populations={'NATURAL99':99,'CLEAN29':29,'WOOD45':45},
            all_frames_retained=True,IoU_filter=False,metric='Unchanged center translation cm; full physical C2 rotation degrees; corner8 SQPnP+LM.',
            natural_joint_medians='All3 seeds strictly lower T and R than paired R0_ONLY, paired original DIVERSE251, R0, PRIOR1 and FULL125.',
            uncertainty='2000 crossed paired recording/seed bootstrap, seed20261001; upper95<0 for both T/R versus paired R0_ONLY and R0.',
            recording_sensitivity='Leave each natural recording out: both mean-seed medians lower versus paired R0_ONLY and R0.',
            natural_tail='Mean-seed T/R P90<=1.05 versus paired R0_ONLY and R0; no seed increases failures.',
            clean='Mean-seed T/R median and P90<=1.05 R0; no seed increases failures.',
            wood='All45 retained as separate form-transfer stress, no selection or dropped seed.',
            historical_gate_parity='Same original stability gate structure and tolerances; R0_ONLY is this intervention matched control. No old trained coordinate outputs are changed.',
            reference_before_selection=False,quality_read_after_choices_locked=True,
            limitations='Repeated DEV and source VAL exposure via frozen refiner; not new independent TEST. Passing alone does not prove unseen-recording generalization.'),
        stop='Any input/code mismatch or partial unreceipted fit stops. Source VAL failure stops this candidate method before real routing; no retuning.',
        goal_complete=False)
    C.save(C.DOC/'TRAIN_PROTOCOL.json',p)
    C.save(C.DOC/'TRAIN_PROTOCOL_SHA.json',C.bind(C.DOC/'TRAIN_PROTOCOL.json'))
    print('TRAIN_PROTOCOL_FROZEN',C.sha(C.DOC/'TRAIN_PROTOCOL.json'),flush=True)


if __name__=='__main__':main()
