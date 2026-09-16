"""Calibration lock before held-out-source evaluation; no model fitting."""
import csv, math
import numpy as np
import torch
import env as E
from b_model import MODES,decode_raw
from metrics import measure,summarize,contrast
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset

def load(split):
    return [torch.load(p,map_location='cpu',weights_only=False) for p in sorted((E.RAW/f'B/cache/{split}').glob('*.pt'))]
def predict(c,seed,arm,eta,selection,shuffle=None):
    if eta==0:return c['P'][seed]['B0_raw'].numpy().copy()
    output=c['P'][seed]['output'];T=selection['temperatures'][str(seed)]['temperature']
    bias=shuffle if arm=='B4_SHUFFLE_DIAG' else c['P'][seed]['biases'].get(arm,torch.zeros_like(output['logits'][0]))
    raw,_=decode_raw(output,bias,T,eta,selection['selected_rule'],c['gain'],c['raw_hw'],c['base_raw'])
    return raw.numpy()
def score(c,pred,t):
    return measure(pred,t['target_points'].numpy(),t['target_valid'].numpy(),np.array(c['perms']),c['raw_hw'],c['point_valid'].numpy())
def csvout(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
    torch.set_num_threads(4)
    assert E.read(E.DOC/'B/ADAPTER_TESTS.json')['PASS']
    assert E.read(E.DOC/'B/SHUFFLE_COMPLETE.json')['complete']
    selection=E.read(E.C.B/'P_SELECTION.json');eta_grid=E.read(E.DOC/'PROTOCOL_LOCK.json')['B']['eta_grid']
    caches=load('calibration');ds=ObservationDataset(E.EXPORT/'calibration.json',targets=True)
    target=[ds[i] for i in range(len(ds))];rows=[];selected={}
    for arm in MODES:
        costs=[]
        for eta in eta_grid:
            scores=[]
            for seed in [1,2,3]:
                preds=[predict(c,seed,arm,eta,selection) for c in caches]
                # Predictions serialized first, only then measured against separate supervision.
                path=E.RAW/f'B/calibration_predictions/{arm}_{eta}_seed{seed}.pt';path.parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(ids=[c['frame_id'] for c in caches],points=np.array(preds),GT_input=False),path)
                metrics=[score(c,p,t) for c,p,t in zip(caches,preds,target)];s=summarize(metrics)
                scores.append(s['primary_E_sym']);rows.append(dict(arm=arm,eta=eta,seed=seed,**{k:v for k,v in s.items() if not isinstance(v,dict)}))
            costs.append(float(np.mean(scores)))
        best=next(j for j,v in enumerate(costs) if v<=min(costs)+1e-12)
        selected[arm]=eta_grid[best]
        print('SELECT',arm,selected[arm],costs,flush=True)
    csvout(E.DOC/'B/calibration_all_eta.csv',rows)
    E.freeze(E.DOC/'B/selection_lock.json',dict(complete=True,selected_eta=selected,selection_population='calibration256',
      no_real_selection=True,grid=eta_grid,primary='mean three seed all-frame E_sym',
      results=E.bound(E.DOC/'B/calibration_all_eta.csv'),original_P_selection=E.bound(E.C.B/'P_SELECTION.json')))
    caches=load('synth_val');ds=ObservationDataset(E.EXPORT/'synth_val.json',targets=True)
    target=[ds[i] for i in range(len(ds))];stores={};summary={};flat=[];damage={}
    for arm in ['B0_P',*MODES,'B4_SHUFFLE_DIAG']:
        eta=0 if arm=='B0_P' else selected['B3_THREE_AMBIG'] if arm=='B4_SHUFFLE_DIAG' else selected[arm]
        stores[arm]={};summary[arm]={}
        for seed in [1,2,3]:
            preds=[]
            for i,c in enumerate(caches):
                shuffle=torch.load(E.RAW/f'B/shuffle/synth_val/{i:04d}.pt',map_location='cpu',weights_only=False)['bias'] if arm=='B4_SHUFFLE_DIAG' else None
                preds.append(predict(c,seed,arm,eta,selection,shuffle))
            path=E.RAW/f'B/synth_val_predictions/{arm}_seed{seed}.pt';path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(ids=[c['frame_id'] for c in caches],points=np.array(preds),GT_input=False),path)
            allmetrics=[dict(frame_id=c['frame_id'],**score(c,p,t)) for c,p,t in zip(caches,preds,target)]
            metrics=[r for r in allmetrics if r['evaluable']]
            stores[arm][seed]=metrics;summary[arm][seed]=dict(**summarize(metrics),
              no_annotation_frame_ids=[r['frame_id'] for r in allmetrics if not r['evaluable']])
            if arm!='B0_P':assert [r['frame_id'] for r in metrics]==[r['frame_id'] for r in stores['B0_P'][seed]]
            for r in metrics:flat.append(dict(arm=arm,seed=seed,eta=eta,**{k:v for k,v in r.items() if not isinstance(v,list)}))
    pairs={}
    for arm in ['B0_P','B1_ONE_AMBIG','B2_THREE_EQUAL','B4_SHUFFLE_DIAG']:
        pairs['B3-'+arm]=contrast([[r['E_sym'] for r in stores['B3_THREE_AMBIG'][s]] for s in [1,2,3]],
                                  [[r['E_sym'] for r in stores[arm][s]] for s in [1,2,3]])
    for seed in [1,2,3]:
        base=stores['B0_P'][seed];new=stores['B3_THREE_AMBIG'][seed];d=np.array([n['frame_mean_px']-b['frame_mean_px'] for b,n in zip(base,new)])
        # Use each method's single whole-object branch; separately indicate correspondence changes.
        be=np.concatenate([r['errors'] for r in base]);ne=np.concatenate([r['errors'] for r in new]);good=be<5
        damage[seed]=dict(good_baseline_points=int(good.sum()),good5_to_bad10=int((good&(ne>10)).sum()),
          good5_to_bad10_rate=float((good&(ne>10)).sum()/max(good.sum(),1)),
          delta_mean_px=float(d.mean()),delta_median_px=float(np.median(d)),delta_P90_px=float(np.quantile(d,.9)),
          improved_frames=int((d< -1e-9).sum()),unchanged_frames=int((abs(d)<=1e-9).sum()),harmed_frames=int((d>1e-9).sum()),
          evaluation_symmetry_changed_frames=sum(b['branch']!=n['branch'] for b,n in zip(base,new)))
    primary=pairs['B3-B0_P'];lo,hi=primary['CI95']
    verdict='NO_LINE_USE_SELECTED' if selected['B3_THREE_AMBIG']==0 else 'LINE_AUXILIARY_DEV_SIGNAL' if hi<0 and primary['improved_seeds']>=2 else 'LINE_AUXILIARY_HARM' if lo>0 else 'LINE_AUXILIARY_NOT_ESTABLISHED'
    csvout(E.DOC/'B/per_frame_results.csv',flat)
    E.write(E.RAW/'B/SYNTH_FULL_METRICS.json',stores)
    E.write(E.DOC/'B/results_and_intervals.json',dict(status='SYNTH_COMPLETE_DEV_PENDING',verdict=verdict,
      selected_eta=selected,synth_val=dict(summary=summary,contrasts=pairs,damage=damage),
      new_training_updates=0,synthetic_population='Previously exposed backbone/P sources; not network-unseen test.',
      secondary_intervals='exploratory, multiplicity unadjusted',pose='PENDING',runtime='PENDING'))
    print('B SYNTH RESULT',verdict,primary,flush=True)
if __name__=='__main__':main()
