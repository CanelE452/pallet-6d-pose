"""Runtime compatibility only: reproduce historical N2 batch/backend/decoder.

The immutable source generator, model, split and 0.001px parity guard remain
unchanged. Actual execution is explicitly bound to this wrapper and its frozen
correction document in every generated chunk and SOURCE_COMPLETE receipt.
"""
from __future__ import annotations

import gc
from pathlib import Path
import math
import json

import numpy as np
import torch

from . import core as P
from . import generate as G
from scripts.research.pallet_dim_conditioned_p_v1 import refiner as D

CORRECTION = P.DOC / 'N2_RUNTIME_CORRECTION.json'
DIAGNOSTIC = P.DOC / 'DIAGNOSTIC_N2_PARITY.json'
ARCHIVE = G.DCP_RAW / 'logits/N2_DIM_ONLY_seed1.npz'
STATE = {}
CHECKS = {}


def initialization(data):
    if STATE:
        assert STATE['data_identity'] == id(data)
        return
    archive = np.load(ARCHIVE)
    rows = archive['rows']
    assert np.array_equal(rows, data.validation_rows)
    assert archive['logits'].dtype == np.float32
    STATE.update(data_identity=id(data), rows=rows, archived_logits=archive['logits'],
                 archived_support=archive['support'], positions={int(row):i for i,row in enumerate(rows)},
                 archived_predictions={r['id']:r for r in P.read(G.N2_PREDICTIONS)['records']},
                 context_start=None, contexts_computed=0, stress_forwards=0)
    archive.close()


def cpu_output(output):
    return {key:output[key].detach().cpu() for key in
            ('points_raw','logits','point_support','candidate_displacements')}


def forward_historical(head, batch):
    """Restore prior cuDNN settings automatically before returning to Replay."""
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32, 'Replay/global path must retain the locked FP32 policy'
    with torch.backends.cudnn.flags(enabled=True, benchmark=False,
                                    deterministic=False, allow_tf32=True):
        output = cpu_output(D.forward(head,batch))
    assert not torch.backends.cudnn.allow_tf32
    return output


@torch.inference_mode()
def historical_n2(head, data, row, original_points, variant, raw_hw, gain, offset):
    initialization(data)
    position=STATE['positions'][int(row)]
    start=position//16*16
    slot=position-start
    if STATE['context_start'] != start:
        # Exactly the original validation-logit batch membership and final-tail
        # size. Other rows provide no GT and are never selector examples here.
        rr=STATE['rows'][start:start+16]
        batch=data.batch(rr,'N2_DIM_ONLY',device='cuda',supervision=False)
        assert not any(key.startswith('gt') for key in batch)
        STATE.update(context_start=start, batch=batch,
                     clean_output=forward_historical(head,batch))
        STATE['contexts_computed']+=1
    batch=STATE['batch']
    assert int(STATE['rows'][start+slot])==int(row)
    if variant == 0:
        output=STATE['clean_output']
    else:
        assert variant==1
        altered=dict(batch)
        altered['points']=batch['points'].clone()
        input_points=(np.asarray(original_points)*gain+offset).astype(np.float32)
        altered['points'][slot]=torch.as_tensor(input_points,device='cuda')
        output=forward_historical(head,altered)
        STATE['stress_forwards']+=1
    single={key:value[slot:slot+1] for key,value in output.items()}
    if variant == 0:
        logits=single['logits'][0].numpy()
        support=single['point_support'][0].numpy()
        archived=STATE['archived_logits'][position]
        assert logits.tobytes()==archived.tobytes(), ('Historical clean logits must be bit-exact',row,float(np.max(np.abs(logits-archived))))
        np.testing.assert_array_equal(support,STATE['archived_support'][position])
    # G.generate_source assigns .selection on its current n2_candidate. Since
    # this function is installed there, the original assignment populates us.
    selection=historical_n2.selection
    rule=selection['rule'];T=selection['temperatures']['N2_DIM_ONLY_seed1']['temperature']
    fraction=rule['max_move_image_diagonal_fraction']
    cap=None if fraction is None else fraction*math.hypot(*raw_hw)*gain
    decoded=D.decode(single,T,rule['lam'],cap)[0].numpy()
    input_points=single['points_raw'][0].numpy()
    delta=decoded.astype(np.float64)-input_points.astype(np.float64)
    result=np.asarray(original_points,dtype=np.float64).copy()
    changed=np.isfinite(input_points).all(-1)&np.any(delta!=0,axis=-1);changed[8]=False
    if rule['lam'] != 0:
        result[changed]+=delta[changed]/gain
    if variant == 0:
        record=G.row_record(data.base,row)
        saved=np.asarray(G.top(STATE['archived_predictions'][record['id']])['keypoints_xy'],dtype=np.float64)+100
        assert result.tobytes()==saved.tobytes(), ('Historical clean coordinates must be bit-exact',row,float(np.max(np.abs(result-saved))))
        CHECKS[int(row)]=dict(row=int(row),id=record['id'],logits_max_abs=0.,coordinates_max_abs_px=0.)
    assert not torch.backends.cudnn.allow_tf32
    return result


def main():
    P.verify()
    correction=P.read(CORRECTION)
    assert correction['status']=='APPROVED_NUMERICAL_RUNTIME_COMPATIBILITY_ONLY'
    for binding in correction['bindings']:
        P.verify_binding(binding)
    assert correction['parity_tolerance_px']==G.PARITY_ATOL_PX==.001
    original_candidate=G.n2_candidate
    original_bindings=G.source_bindings

    def augmented_bindings():
        return dict(original_bindings(),
                    runtime_compatibility_wrapper=P.bound(Path(__file__)),
                    numerical_runtime_correction=P.bound(CORRECTION),
                    parity_diagnostic=P.bound(DIAGNOSTIC),
                    archived_N2_logits=P.bound(ARCHIVE))

    G.n2_candidate=historical_n2
    G.source_bindings=augmented_bindings
    stats={}
    try:
        result=G.generate_source()
        stats=dict(contexts_computed=STATE.get('contexts_computed',0),
                   clean_rows_verified_this_process=len(CHECKS),
                   stress_forwards_this_process=STATE.get('stress_forwards',0))
    finally:
        G.n2_candidate=original_candidate
        G.source_bindings=original_bindings
        STATE.clear();gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
    receipt_path=P.DOC/'N2_RUNTIME_COMPATIBILITY_COMPLETE.json'
    if receipt_path.exists():
        receipt=P.read(receipt_path)
        P.verify_binding(receipt['source_complete'])
        assert receipt['source_complete']==P.bound(P.DOC/'SOURCE_COMPLETE.json')
        return receipt
    assert result['complete'] and result['clean_N2_parity_count']==1616
    assert result['clean_N2_parity_max_abs_px']==0.
    receipt=dict(complete=True,source_complete=P.bound(P.DOC/'SOURCE_COMPLETE.json'),
                 runtime_correction=P.bound(CORRECTION),wrapper=P.bound(Path(__file__)),
                 all_clean_rows_bit_exact=1616,clean_max_abs_px=0.,
                 original_guard_tolerance_px=.001,original_protocol_unchanged=True,
                 Replay_TF32=False,N2_cudnn_TF32=True,N2_matmul_TF32=False,
                 N2_batch='original validation16 membership including original final tail',
                 N2_decoder='CPU, original fixed rule',
                 no_GT_in_candidate_forward=True,**stats)
    P.freeze(receipt_path,receipt)
    print(json.dumps(dict(stage='N2_RUNTIME_COMPATIBILITY_COMPLETE',**stats,clean_max_abs_px=0.)),flush=True)
    return receipt


if __name__=='__main__':main()
