from context35 import *
from data32 import load_render
from probe34 import load_probe
from functools import lru_cache

@lru_cache(maxsize=4)
def label(path):return read(ROOT/path)

def physical(row):
    n=np.zeros((12,2));rho=np.zeros(12);length=np.zeros(12);segment=np.full(12,-1,int);support=np.zeros((12,2,2));off=np.zeros(12)
    origin=np.array(row['origin']);h,w=row['image_hw'];bb=np.array(row['box'])
    for s in label(row['label'])['segments']:
        uv=np.array(s['uv']);q=uv-origin;ok=np.array(s['visible'])&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)&(q[:,0]>=bb[0])&(q[:,0]<=bb[2])&(q[:,1]>=bb[1])&(q[:,1]<=bb[3])
        if row['mask'] is not None:
            a,b,c,d=row['mask'];ok&=~((uv[:,0]>=a)&(uv[:,0]<c)&(uv[:,1]>=b)&(uv[:,1]<d))
        ll=(np.linalg.norm(np.diff(q,axis=0),axis=-1)*(ok[:-1]&ok[1:])).sum();r=s['role']
        if ok.sum()<3 or ll<8 or ll<=length[r]:continue
        v=q[-1]-q[0];normal=np.array([-v[1],v[0]]);normal/=np.linalg.norm(normal)
        if normal[1]<0 or (normal[1]==0 and normal[0]<0):normal=-normal
        n[r]=normal;rho[r]=normal@q[0];length[r]=ll;segment[r]=s['segment'];support[r]=q[np.where(ok)[0][[0,-1]]];off[r]=s['role_offset_m']
    return n,rho,length,segment,support,off

def main():
    register();write(RUN/'EXTRACTION_PROTOCOL.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in [KIT/'context35.py',KIT/'extract35.py']},GT_model_forward=False))
    m,O=load_render();rows=[r for r in m['rows'] if r['detected']];gtrows=[t for t,r in zip(read(V32.RUN/'TARGETS.json')['rows'],m['rows']) if r['detected']];arrays=[physical(r) for r in rows]
    n,rho,length,seg,support,off=[np.array(x) for x in zip(*arrays)];np.savez_compressed(RUN/'geometry.npz',normal=n,rho=rho,length=length,segment=seg,support=support,offset3d=off,gt=np.array([t['points'] for t in gtrows]),outside=np.array([t['outside'] for t in gtrows]),hidden=np.array([t['hidden'] for t in gtrows]))
    write(RUN/'POPULATION.json',dict(rows=rows,requested_rows=len(m['rows']),missed_rows=[dict(id=r['id'],split=r['split'],condition=r['condition'],source_group=r['source_group']) for r in m['rows'] if not r['detected']],all_GT_valid=True))
    deterministic();OO={k:O[k].cuda() for k in ('features','content','edge','box','image_hw')}
    for seed in SEEDS:
        S=torch.load(V32.RUN/'parents'/f'seed{seed}.pt',weights_only=True,map_location='cpu');S={k:S[k].cuda() for k in ('base','strong')};model=load_probe('deterministic',seed);parts=[]
        with torch.no_grad():
            for i in range(0,len(rows),32):
                out=model(subset(OO,slice(i,i+32)),subset(S,slice(i,i+32)),'line_only');parts.append({k:out[k].cpu().numpy() for k in ('points','base','strong','line_normal','line_offset')}|dict(score=model.last_token[...,46].cpu().numpy()))
        z={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};np.savez_compressed(RUN/f'pred_seed{seed}.npz',**z);print('EXTRACT35 seed',seed,flush=True)
    files=[RUN/'geometry.npz',RUN/'POPULATION.json']+list(RUN.glob('pred_seed*.npz'));write(RUN/'EXTRACTION_COMPLETE.json',dict(complete=True,detected=len(rows),files={str(p.relative_to(ROOT)):sha(p) for p in files},GT_model_forward=False,neural_updates=0))
if __name__=='__main__':main()
