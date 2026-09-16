"""GT-blind whole-frame normalized-ROI donor control, frozen before evaluation."""
import math
import numpy as np
import torch
import env as E
from b_model import FrozenHeads

def main():
    torch.set_num_threads(4);E.gpu();heads=FrozenHeads();rng=np.random.default_rng(20260916)
    plans={}
    for split in ['calibration','synth_val']:
        paths=sorted((E.RAW/f'B/cache/{split}').glob('*.pt'))
        records=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
        buckets=[(c['source'],c['asset'],tuple(np.floor(np.log2(np.maximum(np.array(c['box_raw'][2:]-c['box_raw'][:2]),1))).astype(int))) for c in records]
        plan=[]
        for i,c in enumerate(records):
            pool=[j for j,b in enumerate(buckets) if j!=i and b==buckets[i]];level='source_asset_size'
            if not pool:pool=[j for j,b in enumerate(buckets) if j!=i and b[:2]==buckets[i][:2]];level='source_asset'
            if not pool:pool=[j for j in range(len(records)) if j!=i];level='whole_split_fallback'
            j=int(rng.choice(pool));plan.append(dict(receiver=i,donor=j,receiver_id=c['frame_id'],donor_id=records[j]['frame_id'],bucket=level))
        plans[split]=plan
    E.freeze(E.DOC/'B/SHUFFLE_LOCK.json',dict(seed=20260916,GT_input=False,plans=plans,whole_12_channel=True,normalized_ROI_posterior=True))
    for split,plan in plans.items():
        for row in plan:
            i,j=row['receiver'],row['donor'];dst=E.RAW/f'B/shuffle/{split}/{i:04d}.pt'
            if dst.exists():continue
            c=torch.load(E.RAW/f'B/cache/{split}/{i:04d}.pt',map_location='cpu',weights_only=False)
            d=torch.load(E.RAW/f'B/cache/{split}/{j:04d}.pt',map_location='cpu',weights_only=False)
            # All P seeds use exactly the same locations; score once, reuse across seeds.
            for seed in [2,3]:assert torch.equal(c['P'][1]['output']['candidate_displacements'],c['P'][seed]['output']['candidate_displacements'])
            out={k:v.cuda() if torch.is_tensor(v) else v for k,v in c['P'][1]['output'].items()}
            scores=heads.score(out,d['line_logits'].cuda(),d['line_valid'].cuda(),c['box_raw'].cuda(),
              c['base_points'].cuda(),c['point_valid'].cuda(),c['perms'],c['gain'],torch.tensor(c['offset'],device='cuda',dtype=torch.float32))
            dst.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(bias=scores['biases']['B3_THREE_AMBIG'],alignment=scores['alignment'],donor=row),dst)
            if i%128==0:print('B shuffle',split,i,flush=True)
    E.write(E.DOC/'B/SHUFFLE_COMPLETE.json',dict(complete=True,frames=768,GT_input=False,DHT_new_forwards=0,new_training_updates=0))
if __name__=='__main__':main()
