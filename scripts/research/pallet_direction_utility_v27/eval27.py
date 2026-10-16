"""GT-free final model inference, fixed stability gates and paired evaluation."""
from context27 import *
from eval25 import inputs as inputs25,population as population25,conditions as conditions25
from train27 import point_proposal
from model27 import DirectionUtility27
SPLITS=('prospective','paired_check','synth_val','real','camera')
LEGACY=('stock','base','line_vector','v25_point','v25_line')
METHODS=LEGACY+ARMS+('role_line_off',)

def conditions(split):return ('clean','crop_left20','crop_right20','physical_panel') if split=='camera' else conditions25(split)
def population(split,c):
    if split!='camera':return population25(split,c)
    run=ROOT/'data/pallet/results/pallet_camera_trust_v26';m=read(run/'cache/manifest.json');t=read(run/'TARGETS.json')['rows'];take=[i for i,r in enumerate(m['rows']) if r['condition']==c];return dict(rows=[m['rows'][i] for i in take]),[t[i] for i in take]
def inputs(split,c,seed):
    if split!='camera':return inputs25(split,c,seed)
    run=ROOT/'data/pallet/results/pallet_camera_trust_v26';allm=read(run/'cache/manifest.json');chunks=[torch.load(run/'cache'/p,weights_only=True,map_location='cpu') for p in allm['chunks']];O={k:torch.cat([r[k] for r in chunks]) for k in chunks[0]};r=torch.load(run/'parents'/f'seed{seed}.pt',weights_only=True,map_location='cpu');m,_=population(split,c);ii=torch.tensor([r['observation_index'] for r in m['rows'] if r['detected']]);return m,subset(O,ii),subset(r,ii)
def save(split,c,name,m,O,out):
    folder=RUN/'predictions'/split/c;folder.mkdir(parents=True,exist_ok=True);det=np.array([r['detected'] for r in m['rows']]);data=dict(ids=np.array([r['id'] for r in m['rows']]),detected=det)
    for k,v in dict(original_points=O['points'],**out).items():
        z=v.detach().cpu().numpy();a=np.full((len(det),)+z.shape[1:],np.nan,np.float32);a[det]=z;data[k]=a
    np.savez_compressed(folder/(name+'.npz'),**data)

def inference():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['updates']==24000;write(RUN/'EVAL_PROTOCOL.json',dict(training_sha256=sha(RUN/'TRAINING_COMPLETE.json'),source_sha256=sha(Path(__file__)),no_GT_forward=True,fixed_final2000=True))
    for split in SPLITS:
        for c in conditions(split):
            for seed in SEEDS:
                m,O,r=inputs(split,c,seed);O={k:v.cuda() for k,v in O.items()};r={k:v.cuda() for k,v in r.items()};delta=point_proposal(O,r,seed)
                for arm in LEGACY:
                    if arm=='stock' and seed!=1:continue
                    if arm=='v25_line':
                        if split=='camera':old=np.load(ROOT/'data/pallet/results/pallet_camera_trust_v26/predictions'/f'vector_line_edge_seed{seed}.npz');select=[i for i,k in enumerate(old['ids']) if k.split('::')[-1]==c];points=torch.tensor(old['points'][select][old['detected'][select]],device='cuda:0')
                        else:old=np.load(PREV25.RUN/'predictions'/split/c/f'vector_line_edge_seed{seed}.npz');assert old['ids'].tolist()==[r['id'] for r in m['rows']];points=torch.tensor(old['points'][old['detected']],device='cuda:0')
                    else:points=O['points'] if arm=='stock' else r['base']+(r['proposal_delta'] if arm=='line_vector' else delta if arm=='v25_point' else 0)
                    save(split,c,'stock' if arm=='stock' else f'{arm}_seed{seed}',m,O,dict(points=points))
                for arm in ARMS+('role_line_off',):
                    name=f'{arm}_seed{seed}';path=RUN/'predictions'/split/c/(name+'.npz')
                    if path.exists():continue
                    actual='line_role_utility' if arm=='role_line_off' else arm;model=DirectionUtility27(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'{actual}_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']);parts=[]
                    with torch.inference_mode():
                        for i in range(0,len(delta),32):
                            sl=slice(i,i+32);out=model(subset(O,sl),subset(r,sl),delta[sl],actual,line_off=arm=='role_line_off');assert torch.isfinite(out['points']).all();parts.append({k:out[k].cpu() for k in ('points','base','delta','utility','alpha','weights','candidates','candidate_mask')})
                    save(split,c,name,m,O,{k:torch.cat([p[k] for p in parts]) for k in parts[0]});del model
                del O,r,delta
            print('INFER27',split,c,flush=True)
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,fixed_checkpoints=12,GT_forward=False));record('V27 12최종 모델+기존5방법+선 끄기 대조를 별도합성·실사297·V26카메라 전체에 추론. GT를 forward에 주지 않음.')

def scoring():
    assert read(RUN/'INFERENCE_COMPLETE.json')['complete'];results={};comparisons={};diagnostics={};bindings={}
    from sklearn.metrics import roc_auc_score
    for split in SPLITS:
        for c in conditions(split):
            m,ts=population(split,c);valid=np.array([t['valid'] for t in ts]);kind=C.P.target_type(ts).numpy();gt=np.array([t['points'] for t in ts]);rows={};arrays={};ds={}
            for arm in METHODS:
                for seed in ([1] if arm=='stock' else SEEDS):
                    name='stock' if arm=='stock' else f'{arm}_seed{seed}';path=RUN/'predictions'/split/c/(name+'.npz');rows[name],arrays[name]=E.score_file(path,m,ts);bindings[str(path.relative_to(ROOT))]=sha(path)
                    if arm not in ARMS:continue
                    p=np.load(path);mask=valid&p['detected'][:,None];before=np.linalg.norm(p['base']-gt,axis=-1);d=p['candidates'];res=gt-p['base'];aa=np.clip((res[:,:,None]*d).sum(-1)/np.maximum((d*d).sum(-1),1e-12),0,1);opt=p['base'][:,:,None]+aa[...,None]*d;oe=np.linalg.norm(opt-gt[:,:,None],axis=-1);cm=p['candidate_mask']==1;oe=np.where(cm,oe,np.inf);oracle=np.minimum(before,oe.min(-1));actual=np.linalg.norm(p['base'][:,:,None]+p['alpha'][...,None]*d-gt[:,:,None],axis=-1);helped=(before[:,:,None]-actual)>.1;keep=mask[:,:,None]&cm;keep[:,:,0]=False;auc=float(roc_auc_score(helped[keep],p['utility'][keep])) if len(set(helped[keep]))>1 else None;dd={}
                    for typ,k in (('all',None),('hidden',1),('outside',2)):
                        sub=mask if k is None else mask&(kind==k)
                        if not sub.any():continue
                        error=arrays[name]['error'];dd[typ]=dict(points=int(sub.sum()),base_px=float(before[sub].mean()),after_px=float(error[sub].mean()),oracle_candidates_px=float(oracle[sub].mean()),new_harm_over1px=int((sub&(error>before+1)).sum()),keep_weight=float(p['weights'][:,:,0][sub].mean()),point_weight=float(p['weights'][:,:,1:4].sum(-1)[sub].mean()),line_normal_weight=float(p['weights'][:,:,4:7].sum(-1)[sub].mean()) if arm!='point_utility' else None,line_pair_weight=float(p['weights'][:,:,7:].sum(-1)[sub].mean()) if arm!='point_utility' else None,utility_help_auc_all_valid_candidates=auc)
                    ds[name]=dd
            comp={}
            if split=='real':
                for a,b in tuple((a,'base') for a in ARMS)+tuple((a,'point_utility') for a in ARMS if a!='point_utility')+(('line_utility','line_distance'),('line_role_utility','line_utility'),('line_role_utility','role_line_off')):
                    v=COMPARE(arrays,rows,a,b,[t['session'] for t in ts],split);v['subgroups']={typ:COMPARE(arrays,rows,a,b,[t['session'] for t in ts],split,mask=valid&(kind==k)) for typ,k in (('hidden',1),('outside',2))};comp[a+'_vs_'+b]=v
            results[split+'/'+c]=rows;comparisons[split+'/'+c]=comp;diagnostics[split+'/'+c]=ds;print('SCORE27',split,c,flush=True)
    gates={};evidence={}
    for arm in ARMS:
        per={}
        for seed in SEEDS:
            a=f'{arm}_seed{seed}';b=f'base_seed{seed}';v=f'line_vector_seed{seed}';rr=results['real/clean'];g=dict(clean_within1=rr[a]['frame_mean_px']<=1.01*rr[b]['frame_mean_px'])
            for c in ('crop10','crop20','crop30'):
                rr=results['real/'+c];g[c+'_within1']=rr[a]['strata']['outside']['mean_px']<=1.01*rr[b]['strata']['outside']['mean_px']
            for c,typ in (('corner20','hidden'),('crop40','outside')):
                rr=results['real/'+c];g[c+'_retain90']=rr[b]['strata'][typ]['mean_px']-rr[a]['strata'][typ]['mean_px']>=.9*(rr[b]['strata'][typ]['mean_px']-rr[v]['strata'][typ]['mean_px'])
            per[str(seed)]=g
        gates[arm]=dict(per_seed=per,all_conditions_pass=all(all(v.values()) for v in per.values()))
        if arm!='point_utility':
            v=comparisons['real/crop20'][arm+'_vs_point_utility']['subgroups']['outside'];evidence[arm]=dict(per_seed_gain_percent=v['per_seed_gain_percent'],ci95=v['ci95'],pass_gate=min(v['per_seed_gain_percent'])>=1 and v['ci95'][1]<0)
    stable=[a for a in ARMS if gates[a]['all_conditions_pass']];accepted=[a for a in stable if a in evidence and evidence[a]['pass_gate']]
    write(RUN/'SUMMARY.json',dict(complete=True,results=results,comparisons=comparisons,diagnostics=diagnostics,gates=gates,line_evidence=evidence,stable_arms=stable,accepted_line_arms=accepted,scope='12 trained modules; frozen YOLO/DHT/point proposer; no real fitting; real297 reusedDEV and synthetic original split, camera renderer probe. No multiplicity-adjusted or new independent real validation claim.'))
    write(RUN/'EVALUATION_BINDINGS.json',dict(files=bindings));record('V27 사전 고정 판정 완료:stable='+str(stable)+', accepted_line='+str(accepted)+'. 결과를 보고 재학습/임계값 조정하지 않음.');print('VERDICT27',stable,accepted,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('inference','score'),required=True);a=p.parse_args();inference() if a.phase=='inference' else scoring()
