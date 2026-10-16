"""Post-primary arithmetic decomposition; no additional fitting."""
from context37 import *
from analyze37 import load_seed


def main():
    write(RUN/'RESIDUAL_PROTOCOL.json',dict(exploratory=True,primary_sha256=sha(RUN/'SUMMARY.json'),purpose='Explain why physical support-distance gain did not translate to point-normal utility. At GT support midpoint m, decompose correction error e=rho-n.g+b into localization rho-n.m, angular lever (aligned_gt_n-n).(g-m), and physical-offset mismatch b-aligned_gt_n.(g-m). Components are correlated; MAEs not additive. No fitting, gates unchanged. Also audit chosen-candidate harmful rates and conditional attainable improvements.',source_sha256=sha(KIT/'residual37.py')))
    p=population();g=p['g'];N=p['N'];roles=p['roles'];ii=np.arange(N)[:,None];rr=np.arange(8)[None];models=read(V35/'OFFSET_CALIBRATION.json')['models'];arrays={};identity=0.;selection={}
    support=g['support'][ii,roles];m=support.mean(-2);gn=g['normal'][ii,roles];lever=g['gt']-m;v=support[...,1,:]-support[...,0,:];ln=np.linalg.norm(v,axis=-1);t=v/np.maximum(ln[...,None],1e-10);extrap=np.maximum(np.abs((lever*t).sum(-1))-ln/2,0);mask=(~p['train'])[:,None]&p['supported']
    for seed in SEEDS:
      z=load_seed(seed,p)
      for arm in ARMS:
        line=z[arm];top=line['score'].argmax(-1);n=line['normal'][ii,np.arange(12)[None],top][ii,roles];rho=line['rho'][ii,np.arange(12)[None],top][ii,roles];sign=np.where((gn*n).sum(-1)<0,-1.,1.);aligned=gn*sign[...,None];b=np.array([[models[f'{j}_{roles[i,j]}']['median'] for j in range(8)] for i in range(N)])*p['diag'][:,None];local=rho-(n*m).sum(-1);angular=((aligned-n)*lever).sum(-1);offset=b-(aligned*lever).sum(-1);error=rho-(n*g['gt']).sum(-1)+b;identity=max(identity,float(np.abs((local+angular+offset-error)[p['supported']]).max()));arrays[(arm,seed)]=dict(local=local,angular=angular,offset=offset,error=error,extrapolation=extrap,angle_deg=np.degrees(np.arccos(np.clip(np.abs((gn*n).sum(-1)),0,1))))
        for an in ANCHORS:
            a=np.load(RUN/'arrays'/f'{an}_{arm}_seed{seed}.npz');base=np.linalg.norm(a['anchor']-g['gt'],axis=-1);oracle=np.linalg.norm(a['free_point_oracle']-g['gt'],axis=-1)
            for mode in MODES:
                pred=np.linalg.norm(a['selector_'+mode]-g['gt'],axis=-1);selected=a['selected_selector_'+mode]>=0
                for split,frame in [('train',p['train']),('check',~p['train'])]:
                    take=np.broadcast_to(frame[:,None],base.shape);used=take&selected;gain=base-pred;available=base-oracle;selection[f'{an}/{arm}/{seed}/{mode}/{split}']=dict(points=int(take.sum()),selected=int(used.sum()),selected_fraction=float(used.sum()/take.sum()),selected_harm_over1px_fraction=float((gain[used]<-1).mean()) if used.any() else 0.,selected_help_over1px_fraction=float((gain[used]>1).mean()) if used.any() else 0.,mean_actual_gain_px=float(gain[take].mean()),mean_oracle_gain_px=float(available[take].mean()),captured_gain_ratio=float(gain[take].sum()/available[take].sum()))
    rows={}
    for label,sub in [('supported',mask),('within_support',mask&(extrap<=1)),('extrapolated_over1px',mask&(extrap>1)),('outside',mask&g['outside']),('hidden',mask&g['hidden'])]:
        if not sub.any():
            rows[label]=dict(points=0,reason="No GT-supported points in this stratum");continue
        row={}
        for arm in ARMS:
            vals={k:np.stack([arrays[(arm,s)][k] for s in SEEDS])[:,sub] for k in arrays[(arm,1)]};comp=np.stack([vals[k] for k in ('local','angular','offset')],-1).reshape(-1,3)
            row[arm]=dict(points=int(sub.sum()),mae_px={k:float(np.abs(vals[k]).mean()) for k in ('local','angular','offset','error')},component_second_moment=(comp.T@comp/len(comp)).tolist(),angle_median_deg=float(np.median(vals['angle_deg'])),angle_p90_deg=float(np.quantile(vals['angle_deg'],.9)),extrapolation_median_px=float(np.median(vals['extrapolation'])),extrapolation_p90_px=float(np.quantile(vals['extrapolation'],.9)))
        rows[label]=row
    a=[np.load(RUN/'arrays'/f'v34_{arm}_seed{s}.npz') for arm in ('grid','both') for s in SEEDS];dist=np.array([[a[i*3+j]['physical_distance'] for j in range(3)] for i in range(2)]);err=np.array([[np.abs(a[i*3+j]['error']) for j in range(3)] for i in range(2)]);better=dist[1]<dist[0];bad=err[1]>err[0];sm=np.broadcast_to(mask,better.shape)
    assert identity<1e-8
    write(RUN/'RESIDUAL.json',dict(complete=True,exploratory=True,identity_max_px=identity,components=rows,selection=selection,physical_better_normal_worse=dict(better_samples=int((better&sm).sum()),also_normal_worse_samples=int((better&bad&sm).sum()),fraction=float((better&bad&sm).sum()/(better&sm).sum())),limits='GTphysical aligned normals and support midpoint used for explanation only. Correlated signed terms do not permit additive MAE attribution. Candidate oracle may exploit wrong-role or poorly localized lines. Gain-ridge minimizes normalized squared error, while report uses Euclideanpx; no alternate objective experiment performed.'))
    print('RESIDUAL37',read(RUN/'RESIDUAL.json')['physical_better_normal_worse'],rows['supported'],flush=True)

if __name__=='__main__':main()
