"""Replay, train-fit and same-prediction candidate diagnostics, after the trial."""
from pathlib import Path
import json,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pallet_c4_joint_loss_v1 as exp

def main():
    torch.set_num_threads(2)
    OUT=exp.OUT;assert exp.read(OUT/'SUMMARY.json')['complete']
    base=exp.old.base_model();flow=base.model[-1].flow_model.eval()
    saved=np.load(OUT/'predictions.npz');summary=exp.read(OUT/'SUMMARY.json')
    replay=[];fit={};state_differences={}
    for seed in [1,2,3]:
        states={a:torch.load(OUT/'models'/f'{a}_s{seed}'/'final.pt',weights_only=True) for a in exp.ARMS}
        state_differences[str(seed)]=dict(point_vs_joint_max_abs=max(float(abs(states['point_select'][k]-states['joint_select'][k]).max()) for k in states['point_select']),
            indexed_vs_joint_max_abs=max(float(abs(states['indexed'][k]-states['joint_select'][k]).max()) for k in states['indexed']))
    for name in ['stock']+[f'{a}_s{s}' for s in [1,2,3] for a in exp.ARMS]:
        seed=1 if name=='stock' else int(name[-1]);model=exp.old.OutputAdapter(base.model[-1],seed).eval()
        if name!='stock':model.load_state_dict(torch.load(OUT/'models'/name/'final.pt',weights_only=True))
        fit[name]={}
        for split in ['train','val']:
            z=exp.load(split);rows=exp.read(OUT/f'{split}_records.json');record=[];xy=[]
            with torch.no_grad():
                for start in range(0,len(rows),64):
                    b=exp.old.subset(z,slice(start,start+64));p=model(b,True);jj=exp.objective(p,b,flow,'joint_select');pp=exp.objective(p,b,flow,'point_select')
                    decoded=exp.old.prepared_points(p,b).numpy();xy.extend(decoded)
                    for j,r in enumerate(rows[start:start+64]):
                        target=np.array(r['gt_raw']);mask=np.array(r['valid']);opts=exp.PERMS.numpy() if r['order']==4 else exp.PERMS[:1].numpy()
                        errors=[float(np.linalg.norm(decoded[j,:8]-target[q][:8],axis=-1)[mask[q][:8]].mean()) for q in opts]
                        record.append(dict(population=r['population'],joint_loss=float(jj[0][j]),point_selected_loss=float(pp[0][j]),
                            terms_for_joint=jj[3][j,jj[1][j]].tolist(),choice_diff=bool(jj[1][j]!=pp[1][j]),
                            selected=int(jj[1][j]),spatial_px=min(errors),sigma=float(p[j,:8,-2:].sigmoid().mean())))
            if split=='val':replay.append(float(np.max(np.abs(np.array(xy)-saved[name]))))
            fit[name][split]={}
            for pop in ['C1','C4']:
                rs=[r for r in record if r['population']==pop]
                fit[name][split][pop]=dict(n=len(rs),mean_joint_loss=float(np.mean([r['joint_loss'] for r in rs])),
                    mean_point_selected_loss=float(np.mean([r['point_selected_loss'] for r in rs])),
                    terms_for_joint=np.mean([r['terms_for_joint'] for r in rs],axis=0).tolist(),
                    choice_disagreements=sum(r['choice_diff'] for r in rs),nonidentity=sum(r['selected']!=0 for r in rs),
                    spatial_mean_px=float(np.mean([r['spatial_px'] for r in rs])),sigma_mean=float(np.mean([r['sigma'] for r in rs])))
    assert max(replay)<.001,max(replay)
    preserved=exp.read(OUT/'PROTOCOL.json')['sources']
    for path,digest in preserved.items():assert exp.sha(exp.ROOT/path)==digest,path
    for split in ['train','val']:
        assert exp.sha(OUT/f'{split}.pt')==exp.read(OUT/'CACHE_COMPLETE.json')['cache_sha256'][split]
        for r in exp.read(OUT/f'real_{split}_manifest.json'):
            assert exp.sha(exp.ROOT/r['image'])==r['image_sha256']
            assert exp.sha(exp.ROOT/r['label'])==r['label_sha256']
    # Independently replay metrics from saved predictions and immutable records.
    rows=exp.read(OUT/'val_records.json');frame=exp.read(OUT/'PER_FRAME.json')
    from evaluate import iou
    errors=[]
    for name,xy in saved.items():
        for i,r in enumerate(rows):
            gt=np.array(r['gt_raw']);valid=np.array(r['valid']);ps=exp.PERMS.numpy() if r['order']==4 else exp.PERMS[:1].numpy()
            candidates=[np.linalg.norm(xy[i,:8]-gt[p][:8],axis=-1)[valid[p][:8]].mean() for p in ps]
            match=r['confidence']>=.25 and iou(r['box_raw'],r['gt_box_raw'])>=.5
            val=float(min(candidates)) if match else float(np.linalg.norm(r['raw_hw']))
            errors.append(abs(val-frame[name][i]['px']))
    assert max(errors)<1e-7
    exp.write(OUT/'DIAGNOSIS.json',dict(complete=True,fit=fit,state_differences=state_differences,
        training_history=exp.read(OUT/'TRAIN_COMPLETE.json')['cells'],
        scope='post-hoc diagnostics, no extra updates; train uses TAL anchors and val uses GT-free top detections, so train/val fit is not an identical-anchor comparison'))
    exp.write(OUT/'FINAL_AUDIT.json',dict(PASS=True,model_replay_max_px=max(replay),metric_replay_max_px=max(errors),
        C1_initial_loss_gradient_exact=exp.read(OUT/'PREFLIGHT.json')['C1_loss_and_gradient_exact'],
        sources_and_labels_unchanged=True,models=9,updates=9000,state_differences=state_differences))
    print(json.dumps(exp.read(OUT/'FINAL_AUDIT.json'),indent=2))

if __name__=='__main__':main()
