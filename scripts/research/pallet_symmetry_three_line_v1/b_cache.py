"""Extract unchanged source P candidates and native DHT posteriors without GT."""
import time, math
import numpy as np
import torch
import env as E
from b_model import FrozenHeads, decode_raw
from audit_math import line_candidate_evidence
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset,collate,observation_batch
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.export import letterbox,input_to_raw,sample_roi

def cpu(v):
    if torch.is_tensor(v):return v.detach().cpu()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    return v
def main():
    torch.set_num_threads(4);assert torch.cuda.is_available();print(E.gpu(),flush=True)
    heads=FrozenHeads();source=E.read(E.C.LINE/'SOURCE_MANIFEST.json')['records']
    keys=['p3','p4','points','boxes','point_valid','input_shape','gain']
    arrays={k:np.load(E.C.LINE/f'cache/{k}.npy',mmap_mode='r') for k in keys}
    start=time.monotonic();audit=[];peak=0
    for split in ['calibration','synth_val']:
        dataset=ObservationDataset(E.EXPORT/(split+'.json'),targets=False)
        for i,record in enumerate(dataset.records):
            dst=E.RAW/f'B/cache/{split}/{i:04d}.pt';dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():continue
            item=dataset[i];obs=observation_batch(collate([item]),torch.device('cuda'))
            row=record['stock_cache_index'];source_row=source[row]
            assert record['frame_id']==source_row['id'] and record['source_image_sha256']==source_row['image_sha256']
            gain,offset=letterbox(source_row['prepared_shape_hw'],arrays['input_shape'][row])
            assert abs(gain-float(arrays['gain'][row]))<1e-12
            base_raw=input_to_raw(np.array(arrays['points'][row]),gain,offset)
            assert np.allclose(base_raw,item['base_points'].numpy(),atol=1e-4,rtol=0)
            batch={k:torch.tensor(np.array(arrays[k][row]),device='cuda')[None] for k in keys if k!='gain'}
            with torch.no_grad():
                logits,valid=heads.observe(obs)
                saved=dict(frame_id=record['frame_id'],source_image_sha256=record['source_image_sha256'],
                  line_logits=logits[0].cpu(),line_valid=valid[0].cpu(),box_raw=item['box'],
                  base_raw=torch.tensor(base_raw),base_points=item['base_points'],point_valid=item['point_valid'],
                  raw_hw=source_row['raw_shape_hw'],gain=gain,offset=offset,perms=record['symmetry_permutations'],
                  source=record['source'],asset=record['asset'],P={})
                for seed,head in heads.points.items():
                    out=head(*(batch[k] for k in ['p3','p4','points','boxes','point_valid','input_shape']),lam=0)
                    scores=heads.score(out,logits[0],valid[0],obs['box'][0],obs['base_points'][0],obs['point_valid'][0],
                      record['symmetry_permutations'],gain,torch.tensor(offset,device='cuda',dtype=torch.float32))
                    reduced={k:v for k,v in out.items() if k not in ['coverage']}
                    saved['P'][seed]=dict(output=cpu(reduced),**scores)
                    T=heads.selection['temperatures'][str(seed)]['temperature'];rule=heads.selection['selected_rule']
                    baseline,moved=decode_raw(out,scores['biases']['B3_THREE_AMBIG'],T,0,rule,gain,saved['raw_hw'],saved['base_raw'])
                    from generic_point_refiner import decode
                    cap=rule['max_move_image_diagonal_fraction']*math.hypot(*saved['raw_hw'])*gain
                    assert torch.equal(moved,decode(out,T,rule['lam'],cap))
                    assert torch.equal(baseline[8],saved['base_raw'][8])
                    saved['P'][seed]['B0_raw']=baseline
                if i==0:
                    # Independent reconstruction of native source observation from P3/P4.
                    p3,c3=sample_roi(arrays['p3'][row],np.array(item['box']),gain,offset,8,tuple(saved['raw_hw']))
                    p4,c4=sample_roi(arrays['p4'][row],np.array(item['box']),gain,offset,16,tuple(saved['raw_hw']))
                    feature=torch.cat([p3,p4],1)*(c3&c4)
                    delta=float((feature[0]-item['features']).abs().max())
                    # Export used float64 raw box before casting. Tolerance handles its saved float32 boundary.
                    assert delta<5e-4,delta
                    audit.append(dict(split=split,frame_id=record['frame_id'],observation_feature_max_abs=delta,
                       line_forward='stem -> line_head only',utility_or_WLS_calls=0,eta0_native_decode_bit_exact=True))
            torch.save(saved,dst)
            peak=max(peak,torch.cuda.max_memory_allocated())
            if i%32==0:print('B cache',split,i,len(dataset),'seconds',round(time.monotonic()-start,1),flush=True)
    E.write(E.DOC/'B/CACHE_COMPLETE.json',dict(complete=True,frames=768,P_forwards=2304,DHT_observation_forwards=768,
      R0_new_forwards=0,R0_cached_source_frames=768,GT_loaded=False,optimizer_updates=0,peak_allocated_bytes=peak,
      elapsed_seconds=time.monotonic()-start,source_parity=audit,
      files=[E.bound(p) for p in sorted((E.RAW/'B/cache').glob('*/*.pt'))]))
if __name__=='__main__':main()
