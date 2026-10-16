"""GT-free directional adaptation, counterfactual checks and paired population scores."""
from context30 import *
from data30 import evaluation_data,population,conditions
from model30 import DirectionalAdaptive,ANCHORS

BASELINES=('stock','base','strong')
INTERVENTIONS=('normal_no_lines','normal_evidence_off','normal_evidence_donor')
ORACLES=('oracle_anchors_GT','oracle_point_amount_GT','oracle_normal_choice_GT','oracle_normal_amount_GT')
LEGACY=('line_vector','line_role_utility')

def inference():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['updates']==24000
    write(RUN/'EVAL_PROTOCOL.json',dict(source_sha256=sha(Path(__file__)),training_sha256=sha(RUN/'TRAINING_COMPLETE.json'),no_GT_no_state_forward=True))
    for seed in SEEDS:
        models={}
        for arm in ARMS:
            m=DirectionalAdaptive(seed).cuda().eval();m.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',map_location='cpu',weights_only=True)['state_dict']);models[arm]=m
        for split in EVAL_SPLITS:
            for c in conditions(split):
                folder=RUN/'predictions'/split/c;marker=folder/f'DONE_seed{seed}.json'
                if marker.exists():continue
                manifest,O,S,r=evaluation_data(split,c,seed);del r;O={k:v.cuda() for k,v in O.items()};S={k:v.cuda() for k,v in S.items()}
                for arm,points in [('stock',O['points']),('base',S['base']),('strong',S['strong'])]:save(split,c,f'{arm}_seed{seed}',manifest,O,dict(points=points))
                for arm in ARMS+INTERVENTIONS:
                    if arm in INTERVENTIONS and split!='real':continue
                    actual=arm if arm in ARMS else 'normal_gain';parts=[]
                    with torch.no_grad():
                        for i in range(0,len(O['box']),32):
                            sl=slice(i,i+32);mode='off' if arm=='normal_evidence_off' else 'donor' if arm=='normal_evidence_donor' else 'normal'
                            out=models[actual](subset(O,sl),subset(S,sl),actual,evidence_mode=mode)
                            if arm=='normal_no_lines':
                                idx=out['gain'][:,:,:8].argmax(-1);out['chosen']=idx;out['points']=out['candidate_points'].gather(2,idx[...,None,None].expand(-1,-1,1,2))[:,:,0]
                            keys=('points','alpha','gain','chosen','candidate_points','anchors','directions','candidate_mask','diag','base','strong') if arm in ARMS else ('points','chosen')
                            assert torch.isfinite(out['points']).all();parts.append({k:out[k].cpu() for k in keys})
                    save(split,c,f'{arm}_seed{seed}',manifest,O,{k:torch.cat([p[k] for p in parts]) for k in parts[0]})
                write(marker,dict(complete=True,GT_forward=False,state_forward=False));print('INFER30',seed,split,c,flush=True)
        del models
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,GT_forward=False,state_forward=False))

def oracle(p,gt):
    d=p['directions'].astype(float);anchors=p['anchors'].astype(float);res=gt[:,:,None]-anchors
    alpha=np.clip((res*d).sum(-1)/np.maximum((d*d).sum(-1),1e-10),-1,1)
    endpoints=anchors+alpha[...,None]*d;mask=p['candidate_mask']==1
    def choose(points,allowed):
        error=np.linalg.norm(points-gt[:,:,None],axis=-1);error=np.where(allowed,error,np.inf);i=error.argmin(-1)
        return np.take_along_axis(points,i[...,None,None],2)[:,:,0]
    keep=mask.copy();keep[:,:,2:]=False;point=mask.copy();point[:,:,8:]=False
    return dict(oracle_anchors_GT=choose(anchors,keep),oracle_point_amount_GT=choose(endpoints,point),oracle_normal_choice_GT=choose(p['candidate_points'].astype(float),mask),oracle_normal_amount_GT=choose(endpoints,mask))

def score():
    assert read(RUN/'INFERENCE_COMPLETE.json')['complete'];results={};comparisons={};diagnostics={};bindings={}
    for split in EVAL_SPLITS:
        for c in conditions(split):
            m,ts=population(split,c);gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);kind=C.P.target_type(ts).numpy();folder=RUN/'predictions'/split/c;rows={};arrays={};diag={}
            for seed in SEEDS:
                normal=np.load(folder/f'normal_gain_seed{seed}.npz')
                for name,points in oracle(normal,gt).items():np.savez_compressed(folder/f'{name}_seed{seed}.npz',ids=normal['ids'],detected=normal['detected'],points=points.astype(np.float32),original_points=normal['original_points'],GT_privileged=np.array(True))
                for arm in BASELINES+ARMS+ORACLES+LEGACY+(INTERVENTIONS if split=='real' else ()):
                    path=(OLD27.RUN if arm in LEGACY else RUN)/'predictions'/split/c/f'{arm}_seed{seed}.npz';name=f'{arm}_seed{seed}';rows[name],arrays[name]=E.score_file(path,m,ts);bindings[str(path.relative_to(ROOT))]=sha(path)
                aa={}
                for arm in ARMS:
                    p=np.load(folder/f'{arm}_seed{seed}.npz');error=arrays[f'{arm}_seed{seed}']['error'];before=np.linalg.norm(p['base']-gt,axis=-1);strong=np.linalg.norm(p['strong']-gt,axis=-1);mask=valid&p['detected'][:,None];ii=np.nan_to_num(p['chosen']).astype(int);aidx=np.array(ANCHORS)[ii];anchor=np.where(aidx[...,None]==0,p['base'],p['strong']);anchor_error=np.linalg.norm(anchor-gt,axis=-1)
                    chosen_amount=np.take_along_axis(p['alpha'],ii[...,None],-1)[...,0];dd={}
                    for typ,k in (('all',None),('intact',0),('hidden',1),('outside',2)):
                        sub=mask if k is None else mask&(kind==k)
                        if not sub.any():continue
                        line=sub&(ii>=8);moving=np.linalg.norm(p['points']-anchor,axis=-1)>.1;hard=sub&(before>strong+1)
                        gain_target=(before[:,:,None]-np.linalg.norm(p['candidate_points']-gt[:,:,None],axis=-1))/p['diag'][:,None,None]
                        alternative=(p['candidate_mask']==1)&sub[:,:,None];alternative[:,:,0]=False
                        predpos=p['gain']>0;actualpos=gain_target>0;tp=int((predpos&actualpos&alternative).sum());fp=int((predpos&~actualpos&alternative).sum())
                        dd[typ]=dict(points=int(sub.sum()),mean_px=float(error[sub].mean()),base_px=float(before[sub].mean()),strong_px=float(strong[sub].mean()),keep_base=int((sub&(ii==0)).sum()),keep_strong=int((sub&(ii==1)).sum()),partial_point=int((sub&(ii>=2)&(ii<8)).sum()),line_selected=int(line.sum()),line_improve_anchor_over1px=int((line&(error<anchor_error-1)).sum()),line_harm_anchor_over1px=int((line&(error>anchor_error+1)).sum()),line_signed_amount_negative=int((line&(chosen_amount<0)).sum()),line_moving=int((line&moving).sum()),harm_base_over1px=int((sub&(error>before+1)).sum()),strong_helped_points=int(hard.sum()),strong_helped_gain_retention=float((before[hard]-error[hard]).sum()/(before[hard]-strong[hard]).sum()) if hard.any() else None,predicted_positive_gain_precision=tp/(tp+fp) if tp+fp else None)
                    aa[arm]=dd
                diag[str(seed)]=aa
            arrays['stock']=arrays['stock_seed1'];rows['stock']=rows['stock_seed1'];comp={}
            if split=='real':
                for a,b in [('normal_gain','point_gain'),('normal_gain','whole_line'),('normal_gain','normal_no_evidence'),('normal_gain','normal_no_lines'),('normal_gain','normal_evidence_off'),('normal_gain','normal_evidence_donor')]:
                    v=COMPARE(arrays,rows,a,b,[t['session'] for t in ts],split);v['subgroups']={typ:COMPARE(arrays,rows,a,b,[t['session'] for t in ts],split,mask=valid&(kind==k)) for typ,k in (('hidden',1),('outside',2))};comp[a+'_vs_'+b]=v
            results[split+'/'+c]=rows;comparisons[split+'/'+c]=comp;diagnostics[split+'/'+c]=diag;print('SCORE30',split,c,flush=True)
    gates={}
    for arm in ARMS:
        per={}
        for seed in SEEDS:
            a=f'{arm}_seed{seed}';base=f'base_seed{seed}';old=f'line_vector_seed{seed}';rr=results['real/clean'];g=dict(clean_within1=rr[a]['frame_mean_px']<=1.01*rr[base]['frame_mean_px'])
            for c in ('crop10','crop20','crop30'):
                rr=results['real/'+c];g[c+'_within1']=rr[a]['strata']['outside']['mean_px']<=1.01*rr[base]['strata']['outside']['mean_px']
            for c,t in (('corner20','hidden'),('crop40','outside')):
                rr=results['real/'+c];g[c+'_retain90']=rr[base]['strata'][t]['mean_px']-rr[a]['strata'][t]['mean_px']>=.9*(rr[base]['strata'][t]['mean_px']-rr[old]['strata'][t]['mean_px'])
            per[str(seed)]=g
        gates[arm]=dict(per_seed=per,pass_gate=all(all(v.values()) for v in per.values()))
    v=comparisons['real/crop20']['normal_gain_vs_point_gain']['subgroups']['outside'];evidence=dict(per_seed_gain_percent=v['per_seed_gain_percent'],ci95=v['ci95'],pass_gate=min(v['per_seed_gain_percent'])>=1 and v['ci95'][1]<0)
    accepted=gates['normal_gain']['pass_gate'] and evidence['pass_gate'];write(RUN/'SUMMARY.json',dict(complete=True,accepted=accepted,results=results,comparisons=comparisons,diagnostics=diagnostics,gates=gates,line_evidence=evidence));write(RUN/'EVALUATION_BINDINGS.json',dict(files=bindings));print('VERDICT30',accepted,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('inference','score'),required=True);a=p.parse_args();inference() if a.phase=='inference' else score()
