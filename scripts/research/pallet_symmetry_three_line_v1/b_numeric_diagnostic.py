"""Investigate pre-performance historical line-coefficient parity failure."""
import torch,numpy as np
import env as E
from b_model import FrozenHeads
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset,collate,observation_batch
from scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.geometry import decode_single_mode
def main():
    E.gpu();torch.set_num_threads(4);h=FrozenHeads()
    ds=ObservationDataset(E.EXPORT/'synth_val.json',targets=False)
    old=E.read(E.ROOT/'data/pallet/results/pallet_symmetry_dht_local_v2_wls_correction/predictions/hough_seed1_synth_val.json')['records'];old={r['frame_id']:r for r in old}
    rows=[]
    for start in range(0,len(ds),4):
        obs=observation_batch(collate([ds[i] for i in range(start,min(start+4,len(ds)))]),torch.device('cuda'))
        logits,valid=h.observe(obs)
        for j in range(len(logits)):
            i=start+j;c=torch.load(E.RAW/f'B/cache/synth_val/{i:04d}.pt',map_location='cpu',weights_only=False)
            stored=c['line_logits'][None].cuda();mask=c['line_valid'][None].cuda();box=c['box_raw'][None].cuda()
            a=decode_single_mode(stored,mask,h.line.lattice,box)
            b=decode_single_mode(logits[j:j+1],valid[j:j+1],h.line.lattice,box)
            orig=old[c['frame_id']];r=np.array(orig['raw_line']);new=a['raw_line'][0].cpu().numpy()
            # Signed incidence at the four raw image extrema is the physically relevant line perturbation.
            hh,ww=c['raw_hw'];pts=np.array([[0,0,1],[ww,0,1],[0,hh,1],[ww,hh,1]])
            incidence=np.max(np.abs((new-r)@pts.T))
            rows.append(dict(frame_id=c['frame_id'],batch1_vs4_logits=float((stored-logits[j:j+1]).abs().max()),
              batch1_vs4_rawline=float((a['raw_line']-b['raw_line']).abs().max()),historical_coefficient=float(np.max(abs(new-r))),
              historical_incidence_px=float(incidence),anchor_same=a['anchor_index'][0].cpu().tolist()==orig['anchor_index']))
    E.write(E.DOC/'B/NUMERIC_DIAGNOSTIC.json',dict(initial_failure=dict(test='CPU decode vs historical GPU line coefficient atol .0005',observed=.00274658203125),
      calibration_not_opened=not (E.DOC/'B/selection_lock.json').exists(),rows=rows,
      max_incidence=max(r['historical_incidence_px'] for r in rows),max_coefficient=max(r['historical_coefficient'] for r in rows),
      same_anchors=sum(r['anchor_same'] for r in rows),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32))
    print('NUMERIC',max(r['historical_incidence_px'] for r in rows),max(r['historical_coefficient'] for r in rows),sum(r['anchor_same'] for r in rows),flush=True)
if __name__=='__main__':main()
