"""Save GT-free predictions first; score fixed experts, routing and privileged bounds."""
from context29 import *
from model29 import *
from eval27 import inputs,population,conditions
from train29 import cache_lines,cached_experts

METHODS=('stock','point','guided','independent')+ROUTERS+('predicted_state','oracle_expert_GT','oracle_state_GT')
LEGACY=('base','line_vector','line_role_utility')

def models(seed):
    point=PointExpert(seed).cuda();ds={};ls={};rs={}
    def restore(m,name):
        m.load_state_dict(torch.load(RUN/'models'/f'{name}_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']);return m.eval()
    restore(point,'point')
    for a in ('guided','independent'):
        ls[a]=restore(LineExpert(V6/f'cache_heads/hough_seed{seed}/checkpoint_final.pt',seed,a).cuda(),a+'_dht')
        ds[a]=restore(LineDecoder(seed).cuda(),a+'_decoder')
    for a in ROUTERS:rs[a]=restore(Router(seed,a).cuda(),a)
    return point,ls,ds,rs

def inference():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['updates']==25200
    write(RUN/'EVAL_PROTOCOL.json',dict(source_sha256=sha(Path(__file__)),training_sha256=sha(RUN/'TRAINING_COMPLETE.json'),no_GT_forward=True,final_only=True))
    checks=[]
    for seed in SEEDS:
        point,ls,ds,rs=models(seed)
        for split in EVAL_SPLITS:
            for c in conditions(split):
                folder=RUN/'predictions'/split/c;done=folder/f'DONE_seed{seed}.json'
                if done.exists():checks.extend(read(done)['checks']);continue
                m,O,r=inputs(split,c,seed);del r;O.pop('edge');O={k:v.cuda() for k,v in O.items()}
                lines={a:cache_lines(ls[a],O) for a in ls};cache=cached_experts(point,ds,O,lines);cand=cache['candidates'];quality=cache['quality']
                for i,a in enumerate(('stock','point','guided','independent')):save_prediction(split,c,f'{a}_seed{seed}',m,cand[:,:,i],original_points=O['points'])
                batchparts=[];lineparts=[]
                with torch.no_grad():
                    for first in range(0,len(cand),32):
                        sl=slice(first,first+32);obs=subset(O,sl);cc=cand[sl];qq=quality[sl];part={}
                        for a in ROUTERS:
                            out=rs[a](obs,cc,qq)
                            for k in ('points','risk','state','chosen'):part[a+'_'+k]=out[k]
                        state=part['adaptive_state'].argmax(-1);part['predicted_state_points']=cc.gather(2,(state+1)[...,None,None].expand(-1,-1,1,2))[:,:,0]
                        batchparts.append(part);lp={}
                        for a in ls:
                            l,mass=lines[a]['lines'][sl],lines[a]['mass'][sl];out=ds[a](l,mass,obs['box']);center,half=geometry(obs)
                            lp[a+'_lines']=roi_lines_to_raw(l,obs['box']);lp[a+'_mass']=mass
                            lp[a+'_candidates']=center[:,None,None]+half[:,None,None]*out['candidates'];lp[a+'_mask']=out['mask']
                        lineparts.append(lp)
                out={k:torch.cat([x[k] for x in batchparts]) for k in batchparts[0]}
                for a in ROUTERS:save_prediction(split,c,f'{a}_seed{seed}',m,out[a+'_points'],original_points=O['points'],risk=out[a+'_risk'],state=out[a+'_state'],chosen=out[a+'_chosen'])
                save_prediction(split,c,f'predicted_state_seed{seed}',m,out['predicted_state_points'],original_points=O['points'])
                lp={k:torch.cat([x[k] for x in lineparts]) for k in lineparts[0]}
                save_prediction(split,c,f'bundle_seed{seed}',m,cand,original_points=O['points'],candidates=cand,quality=quality,diag=diagonal(O),**lp)
                localchecks=[]
                if split=='real':
                    with torch.no_grad():
                        obs=subset(O,slice(0,4));alt=dict(obs,points=obs['points']+1000)
                        x=ls['independent'](obs,True);y=ls['independent'](alt,True)
                        assert torch.equal(x['logits'],y['logits'])
                        assert torch.equal(ds['independent'](x['lines'],x['mass'],obs['box'])['points'],ds['independent'](y['lines'],y['mass'],obs['box'])['points'])
                    # Intervention only: original trained guided DHT receives zero or
                    # another observation's normalized point field; weights unchanged.
                    ls['guided'].mode='independent';zero=cache_lines(ls['guided'],O);ls['guided'].mode='guided'
                    bb=O['box'];wh=(bb[:,2:]-bb[:,:2]).clamp_min(1);norm=(O['points']-bb[:,None,:2])/wh[:,None]
                    donor=torch.roll(torch.arange(len(bb),device='cuda:0'),1);donorO=dict(O,points=bb[:,None,:2]+norm[donor]*wh[:,None]);dl=cache_lines(ls['guided'],donorO)
                    for a,ll in (('guided_zero',zero),('guided_donor',dl)):
                        parts=[]
                        with torch.no_grad():
                            for i in range(0,len(bb),32):parts.append(ds['guided'](ll['lines'][i:i+32],ll['mass'][i:i+32],bb[i:i+32])['points'])
                        save_prediction(split,c,f'{a}_seed{seed}',m,torch.cat(parts),original_points=O['points'])
                    localchecks.append(dict(seed=seed,split=split,condition=c,independent_point_shift_invariant=True,donor_index_sha256=digest(donor.cpu().numpy().tobytes().hex())))
                write(done,dict(complete=True,checks=localchecks));checks.extend(localchecks)
                del O,lines,cache,cand,out,batchparts,lineparts,lp
                print('INFER29',seed,split,c,flush=True)
        del point,ls,ds,rs
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,GT_forward=False,checks=checks));record('V29 전 분기 및 적응 선택 최종 추론 완료. GT없는 결과 파일을 저장한 뒤 정량/GToracle 분석 진행.')

def score():
    assert read(RUN/'INFERENCE_COMPLETE.json')['complete'];results={};comparisons={};diagnostics={};bindings={}
    # Compatibility-only migration for a worker that started before original_points
    # metadata was added. Derive it from the saved stock output, never from GT.
    migrated=[]
    for split in EVAL_SPLITS:
        for c in conditions(split):
            folder=RUN/'predictions'/split/c
            for seed in SEEDS:
                stock=np.load(folder/f'stock_seed{seed}.npz')['points']
                for path in folder.glob(f'*seed{seed}.npz'):
                    with np.load(path) as z:
                        if 'original_points' in z:continue
                        data={k:z[k] for k in z.files}
                    data['original_points']=stock;np.savez_compressed(path,**data);migrated.append(str(path.relative_to(ROOT)))
    write(RUN/'FORMAT_COMPATIBILITY.json',dict(original_points_added_from_saved_stock=migrated,no_predictions_or_GT_changed=True,scoring_source_sha256=sha(Path(__file__))))
    for split in EVAL_SPLITS:
        for c in conditions(split):
            m,ts=population(split,c);gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);kind=C.P.target_type(ts).numpy();folder=RUN/'predictions'/split/c
            rows={};arrays={};diagn={}
            for seed in SEEDS:
                bp=folder/f'bundle_seed{seed}.npz';b=np.load(bp);assert b['ids'].tolist()==[t['id'] for t in ts]
                cand=b['candidates'];err=np.linalg.norm(cand-gt[:,:,None],axis=-1);best=np.where(np.isfinite(err),err,np.inf).argmin(-1);state=kind+1
                for name,index in (('oracle_expert_GT',best),('oracle_state_GT',state)):
                    pp=np.take_along_axis(cand,index[...,None,None],2)[:,:,0]
                    np.savez_compressed(folder/f'{name}_seed{seed}.npz',ids=b['ids'],detected=b['detected'],points=pp,original_points=b['original_points'],GT_privileged=np.array(True))
                current=METHODS+LEGACY+(('guided_zero','guided_donor') if split=='real' else ())
                for a in current:
                    path=(OLD27.RUN if a in LEGACY else RUN)/'predictions'/split/c/f'{a}_seed{seed}.npz'
                    name=f'{a}_seed{seed}';rows[name],arrays[name]=E.score_file(path,m,ts);bindings[str(path.relative_to(ROOT))]=sha(path)
                mask=valid&b['detected'][:,None];pred=np.load(folder/f'adaptive_seed{seed}.npz');statepred=np.where(np.isfinite(pred['state']),pred['state'],-1e6).argmax(-1)
                dd={}
                for typ,k in (('all',None),('intact',0),('hidden',1),('outside',2)):
                    sub=mask if k is None else mask&(kind==k)
                    if not sub.any():continue
                    route=pred['chosen'][sub].astype(int);actual=arrays[f'adaptive_seed{seed}']['error'];oracle=err.min(-1);oracle_point=err[:,:,:2].min(-1)
                    q=dict(points=int(sub.sum()),state_accuracy=float((statepred[sub]==kind[sub]).mean()),route_counts=np.bincount(route,minlength=4).tolist(),oracle_route_counts=np.bincount(best[sub],minlength=4).tolist(),oracle_expert_px=float(oracle[sub].mean()),oracle_point_px=float(oracle_point[sub].mean()),line_complementary_gain_px=float((oracle_point-oracle)[sub].mean()),routing_regret_px=float((actual-oracle)[sub].mean()),harm_over1px_vs_stock=int((actual[sub]>err[:,:,0][sub]+1).sum()),expert_px=err[sub].mean(0).tolist())
                    for a in ('guided','independent'):
                        le=np.linalg.norm(b[a+'_candidates']-gt[:,:,None],axis=-1);le=np.where(b[a+'_mask']==1,le,np.inf).min(-1)
                        q[a+'_pair_oracle_px']=float(le[sub].mean());q[a+'_pair_available_fraction']=float((b[a+'_mask'][...,:27]==1).any(-1)[sub].mean())
                    confusion=np.zeros((3,3),dtype=int)
                    np.add.at(confusion,(kind[sub],statepred[sub]),1);q['state_confusion']=confusion.tolist();dd[typ]=q
                diagn[str(seed)]=dd;bindings[str(bp.relative_to(ROOT))]=sha(bp)
            comp={}
            arrays['stock']=arrays['stock_seed1'];rows['stock']=rows['stock_seed1']
            if split=='real':
                pairs=[('adaptive','point_router'),('adaptive','no_state'),('independent','guided'),('guided','point'),('guided','guided_zero'),('guided','guided_donor')]
                for a,z in pairs:
                    v=COMPARE(arrays,rows,a,z,[t['session'] for t in ts],split)
                    v['subgroups']={typ:COMPARE(arrays,rows,a,z,[t['session'] for t in ts],split,mask=valid&(kind==k)) for typ,k in (('hidden',1),('outside',2))};comp[a+'_vs_'+z]=v
            results[split+'/'+c]=rows;diagnostics[split+'/'+c]=diagn;comparisons[split+'/'+c]=comp;print('SCORE29',split,c,flush=True)
    gates={};line_evidence={}
    for arm in ROUTERS:
        per={}
        for seed in SEEDS:
            a=f'{arm}_seed{seed}';base=f'base_seed{seed}';old=f'line_vector_seed{seed}';rr=results['real/clean'];g=dict(clean_within1=rr[a]['frame_mean_px']<=1.01*rr[base]['frame_mean_px'])
            for c in ('crop10','crop20','crop30'):
                rr=results['real/'+c];g[c+'_within1']=rr[a]['strata']['outside']['mean_px']<=1.01*rr[base]['strata']['outside']['mean_px']
            for c,t in (('corner20','hidden'),('crop40','outside')):
                rr=results['real/'+c];g[c+'_retain90']=rr[base]['strata'][t]['mean_px']-rr[a]['strata'][t]['mean_px']>=.9*(rr[base]['strata'][t]['mean_px']-rr[old]['strata'][t]['mean_px'])
            per[str(seed)]=g
        gates[arm]=dict(per_seed=per,pass_gate=all(all(v.values()) for v in per.values()))
    v=comparisons['real/crop20']['adaptive_vs_point_router']['subgroups']['outside'];line_evidence=dict(per_seed_gain_percent=v['per_seed_gain_percent'],ci95=v['ci95'],pass_gate=min(v['per_seed_gain_percent'])>=1 and v['ci95'][1]<0)
    accepted=gates['adaptive']['pass_gate'] and line_evidence['pass_gate']
    write(RUN/'SUMMARY.json',dict(complete=True,results=results,comparisons=comparisons,diagnostics=diagnostics,gates=gates,line_evidence=line_evidence,accepted=accepted,scope='3-seed separately fine-tuned experts + original-disjoint routers; reused real DEV synthetic stress. No natural visibility/physical endpoint GT; no deployment generalization claim.'))
    write(RUN/'EVALUATION_BINDINGS.json',dict(files=bindings));print('VERDICT29',accepted,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('inference','score'),required=True);a=p.parse_args();inference() if a.phase=='inference' else score()
