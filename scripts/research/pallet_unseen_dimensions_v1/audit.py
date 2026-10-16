"""Replay critical input, geometry, identity and metric contracts."""
from common import *
from model import Residual
import numpy as np,torch

def main():
    cfg=register();rows=read(OUT/'test_records.json');p=np.load(OUT/'predictions.npz');rr=read(OUT/'PER_FRAME.json')
    z=torch.load(OUT/'test.pt',weights_only=True);torch.set_num_threads(2)
    checks=dict(data_audit=read(OUT/'DATA_AUDIT.json')['PASS'],perfect_GT_translation=read(OUT/'POSE_GT_AUDIT.json')['passed'])
    assert set(z)=={'point_feat','roi_feat','point_coord','roi_coord','visibility','stock','dims'}
    checks['observation_only_keys']=True
    stockmatch=[x['matched'] for x in rr['stock']]
    checks['identical_detection_population']=all([x['matched'] for x in v]==stockmatch for v in rr.values())
    replay={};identity={};init={}
    for seed in cfg['seeds']:
        hashes=[]
        for arm in ARMS:
            m=Residual(arm,seed);hashes.append(hashlib.sha256(b''.join(v.numpy().tobytes() for v in m.state_dict().values())).hexdigest())
            with torch.no_grad():identity[f'{arm}_s{seed}']=float((m({k:v[:4] for k,v in z.items()})-z['stock'][:4]).abs().max())
            name=f'{arm}_s{seed}';m.load_state_dict(torch.load(OUT/'models'/name/'final.pt',weights_only=True));m.eval()
            with torch.no_grad():xy=m(z).numpy()
            replay[name]=float(np.max(np.abs(xy-p[name])))
        init[str(seed)]=len(set(hashes))==1
    checks.update(exact_initial_YOLO_preservation=max(identity.values())==0,identical_parameter_initialization=all(init.values()),
        prediction_replay=max(replay.values())<1e-5,all_finite_predictions=all(np.isfinite(p[k]).all() for k in p.files))
    # Verify exact score arithmetic from predictions without the evaluator's summary helper.
    arithmetic=[];directions={}
    for name in p.files:
        errors=np.linalg.norm(p[name][:,:8]-np.array([r['gt'] for r in rows]),axis=-1)
        for i,r in enumerate(rows):
            expected=errors[i].mean()/np.linalg.norm(r['raw_hw']) if stockmatch[i] else 1.
            arithmetic.append(abs(min(1.,expected)-rr[name][i]['normalized']))
        if name!='stock':
            correction=p[name][:,:8]-p['stock'][:,:8];wanted=np.array([r['gt'] for r in rows])-p['stock'][:,:8]
            mask=np.array(stockmatch)&np.array([not r['seen'] for r in rows])
            e0=np.linalg.norm(wanted,axis=-1).mean(-1);e=errors.mean(-1)
            cosine=(correction*wanted).sum(-1)/(np.linalg.norm(correction,axis=-1)*np.linalg.norm(wanted,axis=-1)+1e-9)
            directions[name]=dict(matched_unseen_frames=int(mask.sum()),fraction_frames_improved=float(np.mean(e[mask]<e0[mask])),
                median_correction_norm_px=float(np.median(np.linalg.norm(correction[mask],axis=-1))),
                mean_direction_cosine=float(cosine[mask].mean()),new_gt20_from_stock_le10=int(((e0<=10)&(e>20)&mask).sum()))
    checks['score_arithmetic']=bool(max(arithmetic)<1e-8)
    result=dict(PASS=all(checks.values()),checks=checks,prediction_replay_max_px=max(replay.values()),
        metric_replay_max=max(arithmetic),direction_diagnostics=directions,
        limits='No independent real sessions. No semantic rotation metric. Scene bootstrap conditional on two meshes.')
    write(OUT/'AUDIT.json',result);assert result['PASS'],checks
    print('Independent replay PASS',checks)
if __name__=='__main__':main()
