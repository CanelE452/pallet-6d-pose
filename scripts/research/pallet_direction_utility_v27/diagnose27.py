"""Training vs evaluation utility observability; no fitting or model selection."""
from context27 import *
from train25 import dataset
from model27 import DirectionUtility27
from sklearn.metrics import roc_auc_score

def main():
    setup();rows=[]
    for seed in SEEDS:
        O,r,gt,valid,ids,kind,n=dataset(seed);O={k:v.cuda() for k,v in O.items()};r={k:v.cuda() for k,v in r.items()};proposal=torch.load(RUN/'train_cache'/f'point_delta_seed{seed}.pt',weights_only=True,map_location='cpu').cuda();target=gt.numpy();valid=valid.numpy();kind=kind.numpy();base=r['base'].cpu().numpy();before=np.linalg.norm(base-target,axis=-1)
        for arm in ARMS:
            model=DirectionUtility27(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']);parts=[]
            with torch.inference_mode():
                for i in range(0,len(ids),32):
                    sl=slice(i,i+32);out=model(subset(O,sl),subset(r,sl),proposal[sl],arm);parts.append({k:out[k].cpu().numpy() for k in ('points','utility','alpha','weights','candidates','candidate_mask')})
            p={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};error=np.linalg.norm(p['points']-target,axis=-1);res=target-base;d=p['candidates'];aa=np.clip((res[:,:,None]*d).sum(-1)/np.maximum((d*d).sum(-1),1e-12),0,1);oracle=np.minimum(before,np.where(p['candidate_mask'],np.linalg.norm(base[:,:,None]+aa[...,None]*d-target[:,:,None],axis=-1),np.inf).min(-1));gain=before[:,:,None]-np.linalg.norm(base[:,:,None]+p['alpha'][...,None]*d-target[:,:,None],axis=-1)
            for c in ('clean','corner20','crop20','crop40','opposite20'):
                cm=np.array([x.endswith('::'+c) for x in ids])[:,None];k=1 if c=='corner20' else 2 if c.startswith('crop') or c=='opposite20' else None;mask=valid&cm
                if k is not None:mask&=kind==k
                cand=mask[:,:,None]&p['candidate_mask'];cand[:,:,0]=False;y=gain[cand]>.1;auc=float(roc_auc_score(y,p['utility'][cand])) if len(set(y))>1 else None
                rows.append(dict(seed=seed,arm=arm,condition=c,points=int(mask.sum()),base_px=float(before[mask].mean()),after_px=float(error[mask].mean()),oracle_candidates_px=float(oracle[mask].mean()),utility_help_auc=auc,keep_weight=float(p['weights'][:,:,0][mask].mean()),normal_weight=float(p['weights'][:,:,4:7].sum(-1)[mask].mean()),pair_weight=float(p['weights'][:,:,7:].sum(-1)[mask].mean())))
            del model
        del O,r,proposal
    write(RUN/'TRAIN_DIAGNOSIS.json',dict(complete=True,rows=rows,scope='Final models replayed on training observations only, not new training. Same GT-derived diagnostics as eval; in-sample frozen parent residuals. Point control normal/pair slots are point-direction candidates, not lines.'));record('V27 학습집합 전체 재추론 진단 완료. 학습된 효용 판별·보정 이득·후보 oracle 여지를 별도 합성/실사 평가와 비교할 수 있게 저장.');print('TRAIN_DIAGNOSIS27',len(rows),flush=True)

if __name__=='__main__':main()
