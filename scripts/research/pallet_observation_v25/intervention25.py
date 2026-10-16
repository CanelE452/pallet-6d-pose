"""Did opposite-side images actually change the training direction distribution?"""
from context25 import *

def main():
    torch.set_num_threads(2);rows=[];paired=[]
    for split in ('new_train','new_check'):
        for seed in SEEDS:
            frames={}
            for variant in ('original','opposite'):
                if variant=='original':m=read(PREV.RUN/'cache'/split/'crop20/manifest.json');ts=new_targets(split,'crop20');r=new_rep(split,'crop20',seed)
                else:m=read(RUN/'cache'/split/'opposite20/manifest.json');ts=read(RUN/'targets'/f'{split}_opposite20.json')['rows'];r=torch.load(RUN/'parents'/split/f'seed{seed}.pt',weights_only=True,map_location='cpu')
                records=[]
                for row,t in zip(m['rows'],ts):
                    if not row['detected']:continue
                    ix=row['observation_index'];b=r['base'][ix].numpy();d=r['proposal_delta'][ix].numpy();res=np.array(t['points'])-b;valid=np.array(t['valid'])&np.array(t['outside']);normal=np.array([1,0]) if row['crop_side']=='left' else np.array([-1,0]);before=np.linalg.norm(res,axis=-1);after=np.linalg.norm(res-d,axis=-1)
                    for p in np.where(valid)[0]:records.append(dict(id=row['id'],point=int(p),base_px=float(before[p]),full_px=float(after[p]),need_inward=bool(res[p]@normal>0),propose_inward=bool(d[p]@normal>0),wrong=bool(res[p]@d[p]<=0)))
                frames[variant]=records;rows.append(dict(split=split,seed=seed,variant=variant,detected=sum(x['detected'] for x in m['rows']),outside_points=len(records),**{k:float(np.mean([x[k] for x in records])) for k in ('base_px','full_px','need_inward','propose_inward','wrong')}))
            old={(r['id'],r['point']):r for r in frames['original']};new={(r['id'],r['point']):r for r in frames['opposite']};common=sorted(set(old)&set(new))
            paired.append(dict(split=split,seed=seed,same_point_outside_both_views=len(common),need_inward_label_changes=sum(old[k]['need_inward']!=new[k]['need_inward'] for k in common),proposal_correctness_changes=sum(old[k]['wrong']!=new[k]['wrong'] for k in common),scope='Same original and point ID, but actual side and parent predictions change. This subset is diagnostic, not training selection.'))
    write(RUN/'INTERVENTION_DIAGNOSIS.json',dict(complete=True,rows=rows,paired=paired,scope='After data intervention, no fitting or sampling based on these values. Inward direction is relative to the retained image. Valid outside-point population changes with crop side; full populations and same-point subsets separated.'))
    for r in rows:print(r,flush=True)
if __name__=='__main__':main()
