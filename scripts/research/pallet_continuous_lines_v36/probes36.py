from context36 import *
from train36 import load_model
from eval36 import predict,save
from data32 import population
from model36 import loss36

def main():
    setup36();write(RUN/'PROBE_PROTOCOL.json',dict(exploratory=True,reason='Refine-only improves realcrop20 while both improves line accuracy but worsens points. Test final inference dependence and geometry/quality competition separately; no new training or changed gates.',interventions=['refine_geometry_donor','refine_refinement_off','refine_line_off','both_geometry_off','both_rank_off'],gradient='Final both checkpoints, first batch of each100-step interval from fixed train plan (20 batches/seed), shared two refiner hidden layers only. Geometry/quality cosine/norm; no optimization. Point loss is detached from refiner.',source_sha256=sha(KIT/'probes36.py')))
    m,ts=population('real','crop20');gradrows=[];scores={};arrays={};folder=RUN/'probes';folder.mkdir(exist_ok=True)
    for seed in SEEDS:
        data=torch.load(RUN/'cache/real/crop20'/f'seed{seed}.pt',weights_only=True,map_location='cpu');x={k:v.cuda() for k,v in data['x'].items()};old=np.load(V34.RUN/'predictions/real/crop20'/f'line_only_r0_seed{seed}.npz')
        for source in ('refine','both'):
            model=load_model(source,seed);out=predict(model,x,source);rows=[r for r in m['rows'] if r['detected']];n=len(rows);ix=[next((i+n//2+j)%n for j in range(n) if rows[(i+n//2+j)%n]['session']!=r['session']) for i,r in enumerate(rows)];donor={k:out[k][ix].cuda() for k in ('normal','offset','loc','feature')}
            modes=[('refine_geometry_donor','refine','geometry_donor'),('refine_refinement_off','grid',None),('refine_line_off','refine','line_off')] if source=='refine' else [('both_geometry_off','rank',None),('both_rank_off','refine',None)]
            for name,arm,intervention in modes:
                result=predict(model,x,arm,intervention,donor if intervention=='geometry_donor' else None);path=folder/f'{name}_seed{seed}.npz';save(result,path,m,old);scores[f'{name}_seed{seed}'],arrays[f'{name}_seed{seed}']=E.score_file(path,m,ts)
            baseline=RUN/'predictions/real/crop20'/f'{source}_seed{seed}.npz';scores[f'{source}_seed{seed}'],arrays[f'{source}_seed{seed}']=E.score_file(baseline,m,ts)
            if source=='both':
                d=torch.load(RUN/'cache'/f'training_seed{seed}.pt',weights_only=True,map_location='cpu');xx={k:v.cuda() for k,v in d['x'].items()};params=list(model.refiner[:4].parameters())
                for batchno in range(0,2000,100):
                    ii=torch.tensor(d['plan'][batchno],device='cuda');batch=subset(xx,ii);out=model(batch,'both');args=[d[k][ii.cpu()].cuda() for k in ('gt','valid','support','hasline')];loss,terms=loss36(out,batch,*args,'both');a=torch.autograd.grad(terms['geometry'],params,retain_graph=True,allow_unused=True);b=torch.autograd.grad(terms['rank'],params,allow_unused=True);a=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,a)]);b=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,b)]);an=float(a.norm());bn=float(b.norm());gradrows.append(dict(seed=seed,batch_step=batchno+1,cosine=float(a@b/(a.norm()*b.norm())) if an*bn>0 else None,geometry_norm=an,rank_norm=bn))
    arrays['stock']=arrays['refine_seed1'];scores['stock']=scores['refine_seed1'];mask=np.array([t['valid'] for t in ts])&np.array([t['outside'] for t in ts]);comp={}
    for name,source in [('refine_geometry_donor','refine'),('refine_refinement_off','refine'),('refine_line_off','refine'),('both_geometry_off','both'),('both_rank_off','both')]:comp[name]=COMPARE(arrays,scores,name,source,[t['session'] for t in ts],'real',mask=mask)
    write(RUN/'PROBES.json',dict(complete=True,exploratory=True,comparisons=comp,gradient_rows=gradrows,metadata_note='For geometry-donor files, normal/offset arrays remain recipient nominal proposal metadata; donor geometry was used only inside point path. These arrays are not used for geometry metrics or visualizations.'));write(RUN/'PROBES_COMPLETE.json',dict(complete=True,files={str(p.relative_to(ROOT)):sha(p) for p in folder.glob('*.npz')}));print('PROBES36 COMPLETE',flush=True)
if __name__=='__main__':main()
