"""Bounded read-only precision diagnostic; never changes the locked experiment."""
import copy
import gc
from pathlib import Path
import json
import numpy as np
import torch
from . import core as P
from . import generate as G
from scripts.research.pallet_dim_conditioned_p_v1.data import PaperData
from scripts.research.pallet_dim_conditioned_p_v1 import refiner as D


@torch.inference_mode()
def run():
    P.setup();P.verify();runtime=[P.gpu()]
    data=PaperData();split=P.read(P.DOC/'SPLIT.json');rows=np.array(split['pool_rows'][:32],dtype=int)
    archived=np.load(G.DCP_RAW/'logits/N2_DIM_ONLY_seed1.npz');positions={int(r):i for i,r in enumerate(archived['rows'])}
    assert np.array_equal(archived['rows'],data.validation_rows)
    selection=P.read(G.DCP_DOC/'CALIBRATION_AND_SELECTION.json');rule=selection['rule'];T=selection['temperatures']['N2_DIM_ONLY_seed1']['temperature']
    old={r['id']:r for r in P.read(G.N2_PREDICTIONS)['records']}
    model=G.load_n2();entries={};checks={}
    for row in rows:
        record=G.row_record(data.base,row);base=G.prepared_prediction(P.read(G.DCP_RAW/f'source_baseline/{row:05d}.json'))
        points=np.array(G.top(base)['keypoints_xy']);gain,offset=P.N.E.old('features').canvas_affine(record['prepared_shape_hw'],data.arrays['input_shape'][row])
        entries[int(row)]=dict(id=record['id'],points=points,gain=gain,offset=offset,old=np.array(G.top(old[record['id']])['keypoints_xy'])+100,raw_hw=record['raw_shape_hw'])
        checks[int(row)]=dict(row=int(row),id=record['id'],tests={})
    def decode_and_compare(output,rr,label):
        for i,row in enumerate(rr):
            if int(row) not in entries:continue
            e=entries[int(row)];pos=positions[int(row)]
            single={k:(v[i:i+1] if torch.is_tensor(v) and v.ndim>0 and len(v)==len(rr) else v) for k,v in output.items()}
            cap=None if rule['max_move_image_diagonal_fraction'] is None else rule['max_move_image_diagonal_fraction']*np.hypot(*e['raw_hw'])*e['gain']
            result={}
            for where in ('cpu','cuda'):
                local={k:v.to(where) if torch.is_tensor(v) else v for k,v in single.items()}
                q=D.decode(local,T,rule['lam'],cap)[0].cpu().numpy()
                inp=local['points_raw'][0].cpu().numpy()
                restored=e['points']+(q.astype(np.float64)-inp.astype(np.float64))/e['gain'];restored[8]=e['points'][8]
                result[where+'_decode_vs_saved_px']=float(np.max(np.abs(restored-e['old'])))
                if where=='cpu':cpu=restored
                else:result['cpu_vs_GPU_decode_px']=float(np.max(np.abs(cpu-restored)))
            result['logits_vs_archive']=float(np.max(np.abs(single['logits'][0].cpu().numpy()-archived['logits'][pos])))
            result['support_exact']=bool(np.array_equal(single['point_support'][0].cpu().numpy(),archived['support'][pos]))
            checks[int(row)]['tests'][label]=result
    try:
        for tf32 in (False,True):
            torch.backends.cudnn.allow_tf32=tf32
            for batch_mode in ('single','original16'):
                batches=([np.array([r]) for r in rows] if batch_mode=='single' else
                         [archived['rows'][start:start+16] for start in sorted({positions[int(r)]//16*16 for r in rows})])
                for rr in batches:
                    b=data.batch(rr,'N2_DIM_ONLY',device='cuda',supervision=False);o=D.forward(model,b)
                    decode_and_compare(o,rr,f'tf32_{tf32}_{batch_mode}')
                    if not tf32 and batch_mode=='single':
                        cache={k:v for k,v in o.items()}
                        cache['logits']=torch.as_tensor(archived['logits'][[positions[int(rr[0])]]],device='cuda')
                        cache['point_support']=torch.as_tensor(archived['support'][[positions[int(rr[0])]]],device='cuda')
                        decode_and_compare(cache,rr,'archived_logits')
        runtime.append(P.gpu())
    finally:
        torch.backends.cudnn.allow_tf32=False
        del model
        gc.collect();torch.cuda.empty_cache()
    summaries={}
    for label in next(iter(checks.values()))['tests']:
        tests=[r['tests'][label] for r in checks.values()]
        summaries[label]={k:max(t[k] for t in tests) for k in tests[0] if k!='support_exact'}
        summaries[label]['support_all_exact']=all(t['support_exact'] for t in tests)
        summaries[label]['rows_above_locked_tolerance']=sum(t['cuda_decode_vs_saved_px']>G.PARITY_ATOL_PX for t in tests)
    result=dict(complete=True,bounded_rows=32,training=False,GT_scoring=False,original_files_modified=False,
                archived_logits_dtype=str(archived['logits'].dtype),archived_logits_shape=list(archived['logits'].shape),
                current_protocol=P.bound(P.DOC/'PROTOCOL.json'),locked_tolerance_px=G.PARITY_ATOL_PX,
                historical_fit_flags={k:P.read(G.DCP_DOC/'fits/N2_DIM_ONLY_seed1.json')[k] for k in ['TF32_matmul','TF32_cudnn','torch_version','cudnn_version']},
                summaries=summaries,rows=list(checks.values()),runtime=runtime,
                bindings=[P.bound(Path(__file__)),P.bound(G.DCP_RAW/'logits/N2_DIM_ONLY_seed1.npz'),P.bound(G.N2_PREDICTIONS)])
    P.freeze(P.DOC/'DIAGNOSTIC_N2_PARITY.json',result)
    print(json.dumps(dict(summaries=summaries,rows=[r for r in checks.values() if r['tests']['tf32_False_single']['cuda_decode_vs_saved_px']>.001]),indent=2),flush=True)
    return result


if __name__=='__main__':run()
