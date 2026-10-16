from context38 import *
from experiment38 import select,wquantile

def main():
    p=population();part=partition(p);g=p['g'];s=read(RUN/'PROBES.json');proto=read(RUN/'PROBE_PROTOCOL.json');assert proto['primary_sha256']==sha(RUN/'SUMMARY.json') and proto['source_sha256']==sha(KIT/'probe38.py');checked=0;curves=0;mask=np.broadcast_to(part['check'][:,None],(p['N'],8))
    for arm in ARMS:
      for seed in SEEDS:
        name=f'{arm}_seed{seed}';a=dict(np.load(V37/'arrays'/f'v34_{arm}_seed{seed}.npz'));v=dict(np.load(RUN/'arrays'/f'{name}.npz'));q=dict(np.load(RUN/'probes'/f'{name}.npz'));base=np.linalg.norm(a['anchor']-g['gt'],axis=-1)
        for oracle,mu,w in [('oracle_mean',v['target'],v['width_geometry_point']),('oracle_radius',v['mu'],np.abs(v['mu']-v['target'])),('oracle_both',v['target'],np.zeros_like(v['mu']))]:
            points,sel,_=select(a,mu,w,p);assert np.array_equal(points,q['points_'+oracle]) and np.array_equal(sel,q['selected_'+oracle]);error=np.hypot(points[...,0]-g['gt'][...,0],points[...,1]-g['gt'][...,1]);assert (error<=base+1e-8).all();assert abs(error[mask].mean()-s['oracles'][name][oracle]['mean_px'])<1e-9;checked+=1
        ratio=np.abs(v['mu']-v['target'])/v['scale_geometry']
        for row in s['curves'][name]:
            rr=ratio if row['family']=='marginal' else ratio.max(-1);qq=wquantile(rr,p['groups'],part['calibration'],row['calibration_level']);assert qq==row['q'];rr=rr.copy();rr[part['check']]=1e9;assert wquantile(rr,p['groups'],part['calibration'],row['calibration_level'])==qq;points,sel,_=select(a,v['mu'],qq*v['scale_geometry'],p);error=np.hypot(points[...,0]-g['gt'][...,0],points[...,1]-g['gt'][...,1]);assert abs(error[mask].mean()-row['check_mean_px'])<1e-9;assert ((sel>=0)&mask).sum()==row['check_selected_points'];curves+=1
    write(RUN/'PROBE_AUDIT.json',dict(status='PASS',oracle_replays=checked,calibration_threshold_curve_replays=curves,check_GT_perturbation_quantile_unchanged=True,primary_unchanged=True,probes_sha256=sha(RUN/'PROBES.json')));print('PROBE AUDIT38 PASS',checked,curves,flush=True)
if __name__=='__main__':main()
