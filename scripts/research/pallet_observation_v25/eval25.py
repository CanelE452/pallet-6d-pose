"""Fixed populations and matched final-checkpoint evaluation, including misses."""
from context25 import *
from prepare25 import ARMS,new_load25
from features25 import edges_for_manifest
from model25 import Correction25
SPLITS=('prospective','paired_check','synth_val','real')
FACTORS=(('gate_paired','gate_original'),('gate_paired','gate_paired_local'),('vector_line','vector_local'),('vector_line_edge','vector_local_edge'),('vector_line_edge','vector_line'),('vector_local_edge','vector_local'),('vector_line','gate_paired'),('vector_local','gate_paired_local'))

def conditions(split):return ('opposite20',) if split=='paired_check' else TRAIN_CONDITIONS if split=='prospective' else CONDITIONS

def inputs(split,c,seed):
    if split=='paired_check':
        m,O=new_load25('new_check');r=torch.load(RUN/'parents/new_check'/f'seed{seed}.pt',weights_only=True,map_location='cpu')
    elif split=='prospective':m,O=new_load('new_check',c);r=new_rep('new_check',c,seed)
    else:m,O,r=eval_inputs(split,c,seed)
    O['edge']=edges_for_manifest(m,('new_check_'+c) if split in ('paired_check','prospective') else split+'_'+c)
    return m,O,r

def population(split,c):
    if split=='paired_check':return read(RUN/'cache/new_check/opposite20/manifest.json'),read(RUN/'targets/new_check_opposite20.json')['rows']
    if split=='prospective':return read(PREV.RUN/'cache/new_check'/c/'manifest.json'),new_targets('new_check',c)
    return manifest(split,c),targets(split,c)

def save25(split,c,name,m,O,out):
    folder=RUN/'predictions'/split/c;folder.mkdir(parents=True,exist_ok=True);det=np.array([r['detected'] for r in m['rows']]);data=dict(ids=np.array([r['id'] for r in m['rows']]),detected=det)
    for k,v in dict(original_points=O['points'],**out).items():
        arr=v.detach().cpu().numpy();z=np.full((len(det),)+arr.shape[1:],np.nan,np.float32);z[det]=arr;data[k]=z
    np.savez_compressed(folder/f'{name}.npz',**data)

def inference():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['complete'];write(RUN/'EVAL_PROTOCOL.json',dict(training_sha256=sha(RUN/'TRAINING_COMPLETE.json'),source_sha256=sha(Path(__file__)),no_GT_forward=True,all_final_models=True))
    for split in SPLITS:
        for c in conditions(split):
            for seed in SEEDS:
                m,O,r=inputs(split,c,seed);O={k:v.cuda() for k,v in O.items()};r={k:v.cuda() for k,v in r.items()}
                for arm in ('stock','base','line_vector'):
                    if arm=='stock' and seed!=1:continue
                    points=O['points'] if arm=='stock' else r['base']+(r['proposal_delta'] if arm=='line_vector' else 0)
                    save25(split,c,'stock' if arm=='stock' else f'{arm}_seed{seed}',m,O,dict(points=points,base=r['base'],proposal_delta=r['proposal_delta'],applied_delta=points-r['base']))
                for arm in ARMS:
                    path=RUN/'predictions'/split/c/f'{arm}_seed{seed}.npz'
                    if path.exists():continue
                    model=Correction25(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']);parts=[]
                    with torch.inference_mode():
                        for start in range(0,len(O['points']),32):
                            sl=slice(start,start+32);out=model(subset(O,sl),subset(r,sl),arm);assert torch.isfinite(out['points']).all();a={k:out[k].cpu() for k in ('points','base','proposal_delta')};a['applied_delta']=(out['points']-out['base']).cpu()
                            if arm.startswith('gate'):a['alpha']=out['alpha'].cpu()
                            parts.append(a)
                    save25(split,c,f'{arm}_seed{seed}',m,O,{k:torch.cat([p[k] for p in parts]) for k in parts[0]});del model
                del O,r
            print('EVAL25',split,c,flush=True)
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,no_GT_forward=True));record('V25 전체 고정 체크포인트 추론 완료. 재사용 실사297/합성505 및 V23확인128, 새로운 반대쪽 잘림 확인128. 검출 실패도 원래 분모에 유지.')

def scoring():
    assert read(RUN/'INFERENCE_COMPLETE.json')['complete'];results={};comparisons={};diagnostics={};bindings={}
    for split in SPLITS:
        for c in conditions(split):
            m,ts=population(split,c);gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);kind=C.P.target_type(ts).numpy();rows={};arrays={};ds={}
            for arm in ('stock','base','line_vector')+ARMS:
                for seed in ([1] if arm=='stock' else SEEDS):
                    name='stock' if arm=='stock' else f'{arm}_seed{seed}';path=RUN/'predictions'/split/c/f'{name}.npz';rows[name],arrays[name]=E.score_file(path,m,ts);bindings[str(path.relative_to(ROOT))]=sha(path)
                    if arm not in ARMS:continue
                    p=np.load(path);assert np.array_equal(p['detected'],arrays['stock']['detected']);res=gt-p['base'];before=np.linalg.norm(res,axis=-1);actual=arrays[name]['error'];d=p['applied_delta'];old=p['proposal_delta'];out={}
                    for typ,k in (('all',None),('hidden',1),('outside',2)):
                        mask=valid&p['detected'][:,None]
                        if k is not None:mask&=kind==k
                        if not mask.any():continue
                        moving=mask&(np.linalg.norm(d,axis=-1)>.1);oldwrong=mask&((res*old).sum(-1)<=0)&(np.linalg.norm(old,axis=-1)>.1)
                        out[typ]=dict(points=int(mask.sum()),base_px=float(before[mask].mean()),actual_px=float(actual[mask].mean()),harm_over1px=int((mask&(actual>before+1)).sum()),base5_to20=int((mask&(before<=5)&(actual>20)).sum()),wrong_moving_fraction=float(((res*d).sum(-1)[moving]<=0).mean()) if moving.any() else None,moving_points=int(moving.sum()),old_wrong_points=int(oldwrong.sum()),old_wrong_corrected_direction=int((oldwrong&((res*d).sum(-1)>0)&(np.linalg.norm(d,axis=-1)>.1)).sum()),old_wrong_mean_change_px=float((actual-before)[oldwrong].mean()) if oldwrong.any() else None)
                    ds[name]=out
            comp={}
            if split=='real':
                sessions=[t['session'] for t in ts]
                for a,b in tuple((a,'base') for a in ARMS)+FACTORS:
                    v=COMPARE(arrays,rows,a,b,sessions,split);v['subgroups']={typ:COMPARE(arrays,rows,a,b,sessions,split,mask=valid&(kind==k)) for typ,k in (('hidden',1),('outside',2))};comp[a+'_vs_'+b]=v
            results[split+'/'+c]=rows;comparisons[split+'/'+c]=comp;diagnostics[split+'/'+c]=ds;print('SCORE25',split,c,flush=True)
    gates={}
    for arm in ARMS:
        per={}
        for seed in SEEDS:
            name=f'{arm}_seed{seed}';b=f'base_seed{seed}';full=f'line_vector_seed{seed}';r=results['real/clean'];g=dict(clean_within1=r[name]['frame_mean_px']<=1.01*r[b]['frame_mean_px'])
            for c in ('crop10','crop20','crop30'):
                r=results['real/'+c];g[c+'_within1']=r[name]['strata']['outside']['mean_px']<=1.01*r[b]['strata']['outside']['mean_px']
            for c,typ in (('corner20','hidden'),('crop40','outside')):
                r=results['real/'+c];g[c+'_retain90']=r[b]['strata'][typ]['mean_px']-r[name]['strata'][typ]['mean_px']>=.9*(r[b]['strata'][typ]['mean_px']-r[full]['strata'][typ]['mean_px'])
            per[str(seed)]=g
        gates[arm]=dict(per_seed=per,all_conditions_pass=all(all(r.values()) for r in per.values()))
    evidence={}
    for a,b in FACTORS:
        rows={}
        for c,typ in (('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside'),('crop10','outside'),('crop30','outside')):
            v=comparisons['real/'+c][a+'_vs_'+b];g=v['relative_gain_percent'] if typ is None else v['subgroups'][typ]['per_seed_gain_percent'];ci=v['paired_px']['ci95'] if typ is None else v['subgroups'][typ]['ci95'];rows[c]=dict(per_seed_gain_percent=g,ci95=ci,pass_gate=min(g)>=1 and ci[1]<0)
        evidence[a+'_vs_'+b]=rows
    write(RUN/'SUMMARY.json',dict(complete=True,results=results,comparisons=comparisons,diagnostics=diagnostics,gates=gates,evidence_gates=evidence,stable_arms=[a for a in ARMS if gates[a]['all_conditions_pass']],scope='All real/synthetic DEV reused. Opposite crop new observations of existing heldout original groups, not new camera views or natural occlusion. Gates and vectors have distinct losses/output families; vector point/line and edge controls matched. No real fitting, no final holdout claim.'))
    write(RUN/'EVALUATION_BINDINGS.json',dict(files=bindings));record('V25 평가/판정 완료: SUMMARY.json의 stable_arms='+str([a for a in ARMS if gates[a]['all_conditions_pass']])+'. 실패 대조 포함 모든 조건·세 시드 기록. 추가 선 이득과 방향 개선을 별도 표로 분리.')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['inference','score'],required=True);a=p.parse_args();inference() if a.phase=='inference' else scoring()
