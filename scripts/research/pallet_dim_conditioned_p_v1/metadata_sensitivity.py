"""Inference-only metadata perturbations. Donors were frozen before evaluation."""
import math
import numpy as np
import torch
import dcp_env as E
from data import PaperData
from refiner import forward,decode,context
from inference import load_head,predict_captured,preservation,serial
from eval_math import measure,summary
from point_inference import replace_selected

def make_context(dims,orders,mode,norm,lock,donor_dims=None):
    dims=np.array(dims);orders=np.array(orders)
    if mode=='D1':dims=np.tile(lock['train_mean_dimensions_WDH'],(len(dims),1))
    if mode=='D2':dims=np.array(donor_dims)
    if mode=='D3':orders=np.array([{1:2,2:4,4:1}[int(g)] for g in orders])
    z=context(dims,orders,norm,True)
    if mode=='D4':z=np.zeros_like(z)
    return z

def dist_metrics(a,b,T):
    a=np.array(a,float)/T;b=np.array(b,float)/T
    def soft(x):
        x=x-x.max(-1,keepdims=True);x=np.exp(x);return x/x.sum(-1,keepdims=True)
    p=soft(a);q=soft(b);mid=(p+q)/2;kl=np.sum(p*np.log((p+1e-30)/(q+1e-30)),-1)
    js=.5*np.sum(p*np.log((p+1e-30)/(mid+1e-30))+q*np.log((q+1e-30)/(mid+1e-30)),-1)
    return dict(KL_mean=float(kl.mean()),JS_mean=float(js.mean()),top1_changed_fraction=float((a.argmax(-1)!=b.argmax(-1)).mean()))

@torch.no_grad()
def main():
    assert E.read(E.DOC/'REAL_DEV_RESULTS.json')['complete'];E.gpu();torch.set_num_threads(4)
    data=PaperData();norm=data.norm;lock=E.read(E.DOC/'METADATA_DIAGNOSTIC_LOCK.json');sel=E.read(E.DOC/'CALIBRATION_AND_SELECTION.json');rule=sel['rule'];held=np.flatnonzero(data.partitions=='heldout')
    byid={r['id']:i for i,r in enumerate(data.source['records'])};donor={r['recipient']:r['donor'] for r in lock['heldout']}
    devcache=E.read(E.DOC/'DEV_CACHE_COMPLETE.json');dev={r['id']:torch.load(E.ROOT/r['path'],map_location='cpu',weights_only=False) for r in devcache['records']};devdonor={r['recipient']:r['donor'] for r in lock['DEV']}
    results={};allrecords={}
    from dev_evaluate import population_metadata,iou
    pe,pop=population_metadata();targets_dev={}
    for item,m in pop:
        t=pe.E._legacy_forbidden_target(item);targets_dev[item.frame_id]=(np.array(t.keypoints_xy),np.array(t.keypoint_supervision_mask),np.array(t.box_xyxy))
    groups=E.read(E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'][0]['permutations']
    for seed in [1,2,3]:
        arm='N4_META_SYM';name=f'{arm}_seed{seed}';head,_=load_head(arm,seed);T=sel['temperatures'][name]['temperature'];d0logs=np.load(E.RAW/f'logits/{name}.npz');index={int(r):i for i,r in enumerate(d0logs['rows'])}
        for split in ['SYNTH_HELDOUT','REAL_DEV']:
            baseline=E.read(E.RAW/f'predictions/{split}/{name}.json')['records'];base={r['id']:r for r in baseline};results.setdefault(split,{});allrecords.setdefault(split,{})
            for mode in ['D1','D2','D3','D4']:
                diagnostics=[];preds=[]
                if split=='SYNTH_HELDOUT':
                    for start in range(0,len(held),16):
                        rows=held[start:start+16];records=[data.source['records'][i] for i in rows];batch=data.batch(rows,arm,supervision=False)
                        dd=np.array([data.side['dimensions'][byid[donor[r['id']]]] for r in records]);batch['context']=torch.from_numpy(make_context(data.side['dimensions'][rows],data.side['order'][rows],mode,norm,lock,dd)).cuda()
                        o=forward(head,batch);gain=data.arrays['gain'][rows];cap=rule['max_move_image_diagonal_fraction']*np.array([math.hypot(*r['raw_shape_hw']) for r in records])*gain if rule['max_move_image_diagonal_fraction'] is not None else None;q=decode(o,T,rule['lam'],cap).cpu().numpy();logs=o['logits'].cpu().numpy()
                        for j,(row,r) in enumerate(zip(rows,records)):
                            original=E.read(E.RAW/f'source_baseline/{row:05d}.json');idx=original['selected_index'];c=replace_selected(original['candidates'],idx,data.arrays['points'][row],q[j],gain[j],rule['lam']);preservation(original['candidates'],c,idx);pred=dict(id=r['id'],selected_index=idx,candidates=serial(c));preds.append(pred)
                            d=dist_metrics(d0logs['logits'][index[int(row)]],logs[j],T);d['id']=r['id'];d['coordinate_delta_px']=float(np.linalg.norm(np.array(c[idx]['keypoints_xy'])[:8]-np.array(base[r['id']]['candidates'][idx]['keypoints_xy'])[:8],axis=-1).mean()) if idx is not None else 0.;diagnostics.append(d)
                else:
                    for fid,row in dev.items():
                        cap=dict(row['captured']);cap.update(p3=cap['p3'].cuda(),p4=cap['p4'].cuda());z=make_context([row['dimensions']],[row['order']],mode,norm,lock,[dev[devdonor[fid]]['dimensions']])
                        pred,diag=predict_captured(head,arm,cap,row['dimensions'],row['order'],T,rule,row['raw_hw'],norm,z);pred['id']=fid;preds.append(pred);idx=pred['selected_index']
                        if diag is not None:
                            old=np.load(E.RAW/f'DEV_logits/{name}/{fid.replace(":","__")}.npz');d=dist_metrics(old['logits'],diag['logits'],T);d['id']=fid;d['coordinate_delta_px']=float(np.linalg.norm(np.array(pred['candidates'][idx]['keypoints_xy'])[:8]-np.array(base[fid]['candidates'][idx]['keypoints_xy'])[:8],axis=-1).mean());diagnostics.append(d)
                E.write(E.RAW/f'metadata_diagnostics/{split}/{name}_{mode}_predictions.json',dict(records=preds,GT_input=False,diagnostic_only=True));scored=[]
                for p in preds:
                    fid=p['id'];idx=p['selected_index'];points=np.full((9,2),np.nan) if idx is None else p['candidates'][idx]['keypoints_xy']
                    if split=='SYNTH_HELDOUT':
                        row=byid[fid];r=data.source['records'][row];kp=np.array(r['targets'][0]['keypoints_normalized']);gt=kp[:,:2]*np.array(r['prepared_shape_hw'])[::-1]-100;valid=kp[:,2]>0;order=int(data.side['order'][row]);perms=data.side['permutations'][row,:order];hw=r['raw_shape_hw'];matched=bool(data.arrays['matched'][row])
                    else:
                        gt,valid,box=targets_dev[fid];perms=groups;hw=dev[fid]['raw_hw'];matched=idx is not None and iou(p['candidates'][idx]['box_xyxy'],box)>=.5
                    m=measure(points,gt,valid,perms,hw,matched,idx is not None);m['id']=fid;scored.append(m)
                s=summary(scored);baseline_summary=E.read(E.DOC/f'{split}_RESULTS.json')['summary'][name]
                result=dict(summary=s,delta_E_sym=s['E_sym']-baseline_summary['E_sym'],**{k:float(np.mean([d[k] for d in diagnostics])) for k in ['KL_mean','JS_mean','top1_changed_fraction','coordinate_delta_px']})
                results[split][name+'_'+mode]=result;allrecords[split][name+'_'+mode]=dict(metrics=scored,diagnostics=diagnostics);print('META_SENSITIVITY',split,seed,mode,result['delta_E_sym'],flush=True)
        del head;torch.cuda.empty_cache()
    E.write(E.RAW/'METADATA_SENSITIVITY_FRAMES.json',allrecords)
    E.write(E.DOC/'METADATA_SENSITIVITY.json',dict(complete=True,results=results,D0='correct metadata results in primary tables',donor_lock=E.bound(E.DOC/'METADATA_DIAGNOSTIC_LOCK.json'),
      no_causal_proof_claim=True,extra_processed_image_model_pairs=4*3*(1985+319),
      extra_neural_head_forward_images=4*3*(1985+sum(r['captured']['selected_index'] is not None for r in dev.values())),extra_backbone_forwards=0,used_for_selection=False))
if __name__=='__main__':main()
