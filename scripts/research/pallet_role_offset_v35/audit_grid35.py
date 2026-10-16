from context35 import *

def main():
    q=read(RUN/'QUANTIZATION.json');assert read(RUN/'QUANTIZATION_PROTOCOL.json')['source_sha256']==sha(KIT/'quantization35.py')
    for p,h in q['files'].items():assert sha(ROOT/p)==h
    rows=read(RUN/'POPULATION.json')['rows'];g=np.load(RUN/'geometry.npz');z=np.load(RUN/'quantization_geometry.npz');a=np.load(RUN/'analysis_seed1.npz');models=read(RUN/'OFFSET_CALIBRATION.json')['models'];check=np.array([r['split']=='check' for r in rows]);mask=check[:,None]&a['supported'];count=0
    for i,r in enumerate(rows):
        b=np.array(r['box']);wh=np.maximum(b[2:]-b[:2],1);mid=(b[:2]+b[2:])/2
        for role in np.where(g['length'][i]>0)[0]:
            k=int(z['index'][i,role]);theta=(k//48)*np.pi/24;rho=(k%48)*2*np.sqrt(2)/47-np.sqrt(2);uv=g['support'][i,role];normalized=2*(uv-mid)/wh;direction=np.array([np.cos(theta),np.sin(theta)]);length=np.linalg.norm(direction*2/wh);expected=np.abs(normalized@direction-rho).mean()/length;assert abs(expected-z['distance'][i,role])<1e-9;count+=1
    for seed in SEEDS:
        p=np.load(RUN/f'quantization_seed{seed}.npz');a=np.load(RUN/f'analysis_seed{seed}.npz');error=np.linalg.norm(p['points']-g['gt'],axis=-1);assert np.isfinite(error).all();assert abs(error[mask].mean()-q['results']['check/all']['per_seed_px'][seed-1])<1e-9
        for i,j in zip(*np.where(mask)):
            role=a['selected_role'][i,j];n=z['normal'][i,role];tangent=np.array([-n[1],n[0]]);assert abs((p['points'][i,j]-a['base'][i,j])@tangent)<1e-8
    write(RUN/'GRID_AUDIT.json',dict(status='PASS',normalized_grid_equations_checked=count,point_error_recomputed=True,tangent_preserved=True,oracle_choice_uses_physical_support_only=True))
if __name__=='__main__':main()
