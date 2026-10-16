"""Separate small post-primary objective controls, no neural training."""
from context37 import *
from analyze37 import load_seed,features,fit,predict,masks,rowstats


def main():
    write(RUN/'OBJECTIVE_PROTOCOL.json',dict(exploratory=True,primary_sha256=sha(RUN/'SUMMARY.json'),reason='Primary gain selector worsens even training points. Distinguish normalized-squared-gain regression from normal-direction regression and exact reported pixel-gain target before attributing its failure to unavailable information.',scope='V34anchor only grid/both3seeds,4samefeaturecontrols,2targets =>48additional ridge fits; same training16/check8groups, same fixednormal .15diag bounded candidates/no-op, same lambda/normalization/groupweight. No threshold tuning.',targets='normal: desired signed normal correction/diag, utility=amount/diag*(2*predicted_normal-amount/diag), exact analytic normalized squared gain. pixel: (anchor Euclideanpx-candidate Euclideanpx)/10, avoiding normalized-square/report-objective mismatch. Choose predictedpositiveutility max else anchor.',gates='Exploratory success requires >=1%eachseed vs anchor, groupCIupper<0, allcondition eachseed<=1.01anchor, geometry+visual beatsvisual CIupper<0. Original gates/results unchanged. Simple linear residual model failure not universal impossibility.',source_sha256=sha(KIT/'objectives37.py')))
    p=population();g=p['g'];inc=np.array(INC);ir=np.repeat(inc,8,axis=1);arrays={};fits=0;(RUN/'objectives').mkdir(exist_ok=True)
    for seed in SEEDS:
      z=load_seed(seed,p)
      for arm in ('grid','both'):
        a=dict(np.load(RUN/'arrays'/f'v34_{arm}_seed{seed}.npz'));anchor=a['anchor'];n=a['candidate_normal'];amount=a['candidate_amount'];candidate=a['candidate'];baseerr=np.linalg.norm(anchor-g['gt'],axis=-1);ce=np.linalg.norm(candidate-g['gt'][:,:,None],axis=-1);vi=z['visual'][:,inc].reshape(p['N'],8,24,32);X=features(p,z,anchor,n,amount,a['candidate_score'],ir,vi)
        targets=dict(normal=(n*(g['gt']-anchor)[:,:,None]).sum(-1)/p['diag'][:,None,None],pixel=(baseerr[...,None]-ce)/10)
        for target,y in targets.items():
            out={};coefs={};metrics={}
            for mode in MODES:
                model=fit(X[mode],y,p['train'],p['groups']);v=predict(model,X[mode]);fits+=1
                if target=='normal':
                    normalized=amount/p['diag'][:,None,None];utility=normalized*(2*v-normalized)
                else:utility=v
                k=utility.argmax(-1);chosen=np.where(utility.max(-1)>0,k,-1);pred=np.where((chosen>=0)[...,None],candidate[np.arange(p['N'])[:,None],np.arange(8)[None],k],anchor);out['points_'+mode]=pred;out['selected_'+mode]=chosen;out['predicted_'+mode]=v
                coefs.update({mode+'_'+key:value for key,value in model.items()});errors=np.linalg.norm(pred-g['gt'],axis=-1);arrays[(target,arm,seed,mode)]=errors
                metrics[mode]={split:dict(mean_target_mae=float(np.abs(v[frame]-y[frame]).mean()),no_prediction_target_mae=float(np.abs(y[frame]).mean()),point_error_px=float(errors[frame].mean()),selected_fraction=float((chosen[frame]>=0).mean())) for split,frame in [('train',p['train']),('check',~p['train'])]}
            np.savez_compressed(RUN/'objectives'/f'{target}_{arm}_seed{seed}.npz',**out);np.savez_compressed(RUN/'objectives'/f'fit_{target}_{arm}_seed{seed}.npz',**coefs);write(RUN/'objectives'/f'{target}_{arm}_seed{seed}.json',metrics)
        print('OBJECTIVE37',seed,arm,'fits',fits,flush=True)
    results={};comp={}
    base=np.stack([np.linalg.norm(np.load(RUN/'arrays'/f'v34_both_seed{k}.npz')['anchor']-g['gt'],axis=-1) for k in SEEDS])
    for key,mask in masks(p):
        row={};cc={}
        for target in ('normal','pixel'):
          for arm in ('grid','both'):
            for mode in MODES:
                err=np.stack([arrays[(target,arm,k,mode)] for k in SEEDS]);row[target+'/'+arm+'/'+mode]=rowstats(err,base,mask,p['groups'])
            err=np.stack([arrays[(target,arm,k,'visual_geometry')] for k in SEEDS]);ref=np.stack([arrays[(target,arm,k,'visual')] for k in SEEDS]);cc[target+'/'+arm+'/geometry_vs_visual']=rowstats(err,ref,mask,p['groups'])
        results[key]=row;comp[key]=cc
    gates={}
    for target in ('normal','pixel'):
        r=results['check/all/all_detected'][target+'/both/visual_geometry'];improve=min(r['gain_percent'])>=1 and r['difference']['ci95'][1]<0;add=comp['check/all/all_detected'][target+'/both/geometry_vs_visual']['difference']['ci95'][1]<0;preserve=all(min(row[target+'/both/visual_geometry']['gain_percent'])>=-1 for k,row in results.items() if k.startswith('check/'));gates[target]=dict(improves=bool(improve),adds_geometry=bool(add),preserves=bool(preserve),passed=bool(improve and add and preserve))
    write(RUN/'OBJECTIVES.json',dict(complete=True,exploratory=True,additional_ridge_fits=fits,total_ridge_fits=96+fits,neural_updates=0,results=results,comparisons=comp,gates=gates));print('OBJECTIVE37 COMPLETE',gates,flush=True)

if __name__=='__main__':main()
