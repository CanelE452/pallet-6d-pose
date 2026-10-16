"""Read-only, stratified direction diagnosis; real GT never fits a model."""
from context25 import *

def main():
    torch.set_num_threads(4)
    RUN.mkdir(parents=True, exist_ok=True)
    records=[];bindings={}
    for split in ('new_train','new_check','real'):
        for c in ('crop20','crop40'):
            mp = PREV.RUN/'cache'/split/c/'manifest.json' if split!='real' else C.V11/'cache'/split/c/'manifest.json'
            tp = PREV.RUN/'targets'/f'{split}_{c}.json' if split!='real' else C.V11/'targets'/f'{split}_{c}.json'
            m=read(mp);ts=read(tp)['rows'];assert [r['id'] for r in m['rows']]==[t['id'] for t in ts]
            bindings[str(mp.relative_to(ROOT))]=sha(mp);bindings[str(tp.relative_to(ROOT))]=sha(tp)
            for seed in SEEDS:
                if split=='real':
                    pp=ROOT/'data/pallet/results/pallet_quality_v23/predictions/real'/c/f'unit_expand_seed{seed}.npz'
                    z=np.load(pp);b=z['base'];d=z['proposal_delta'];assert z['ids'].tolist()==[r['id'] for r in m['rows']]
                else:
                    pp=PREV.RUN/'parents'/split/c/f'seed{seed}.pt';z=torch.load(pp,weights_only=True,map_location='cpu');det=np.array([r['detected'] for r in m['rows']]);b=np.full((len(det),8,2),np.nan);d=b.copy();b[det]=z['base'];d[det]=z['proposal_delta']
                bindings[str(pp.relative_to(ROOT))]=sha(pp)
                for i,(r,t) in enumerate(zip(m['rows'],ts)):
                    if not r['detected']:continue
                    gt=np.asarray(t['points']);res=gt-b[i];before=np.linalg.norm(res,axis=-1);full=np.linalg.norm(res-d[i],axis=-1);move=np.linalg.norm(d[i],axis=-1);dot=(res*d[i]).sum(-1);a=np.clip(dot/np.maximum(move**2,1e-12),0,1)
                    box=np.asarray(r['box']);wh=np.maximum(box[2:]-box[:2],1);diag=np.linalg.norm(wh);h,w=r['image_hw'];side=r['crop_side'];distance=np.maximum.reduce([-b[i,:,0],b[i,:,0]-w,-b[i,:,1],b[i,:,1]-h,np.zeros(8)])/diag
                    normal=np.array([1.,0.]) if side=='left' else np.array([-1.,0.]);inward=res@normal;proposal=d[i]@normal
                    retained=np.prod(np.maximum(np.minimum(box[2:],[w,h])-np.maximum(box[:2],0),0))/np.prod(wh)
                    for p in range(8):
                        if not (t['valid'][p] and t['outside'][p]):continue
                        records.append(dict(split=split,condition=c,seed=seed,id=r['id'],session=r['session'],point=p,side=side,base_px=float(before[p]),full_px=float(full[p]),wrong=bool(dot[p]<=0),alpha=float(a[p]),distance=float(distance[p]),distance_bin=int(np.digitize(distance[p],[.02,.05,.1])),aspect=float(wh[0]/wh[1]),aspect_bin=int(np.digitize(wh[0]/wh[1],[1.5,2.5,4.])),retained=float(retained),retained_bin=int(np.digitize(retained,[.6,.8,.95])),error_bin=int(np.digitize(before[p],np.arange(5,101,5),right=True)),need_inward=bool(inward[p]>0),propose_inward=bool(proposal[p]>0),move_px=float(move[p])))
    table=[]
    for split in ('new_train','new_check','real'):
        for c in ('crop20','crop40'):
            for axis in ('all','side','distance_bin','aspect_bin','retained_bin'):
                pop=[r for r in records if r['split']==split and r['condition']==c]
                for level in ([None] if axis=='all' else sorted({r[axis] for r in pop})):
                    cell=[r for r in pop if level is None or r[axis]==level];per=[]
                    for seed in SEEDS:
                        a=[r for r in cell if r['seed']==seed]
                        if a:per.append(dict(seed=seed,points=len(a),originals=len({r['id'] for r in a}),sessions=len({r['session'] for r in a}),**{k:float(np.mean([r[k] for r in a])) for k in ('base_px','full_px','wrong','alpha','need_inward','propose_inward')}))
                    table.append(dict(split=split,condition=c,axis=axis,level=level,per_seed=per))
    matched=[]
    for c in ('crop20','crop40'):
        for axis in ('none','side','distance_bin','aspect_bin','retained_bin'):
            for seed in SEEDS:
                tr=[r for r in records if r['split']=='new_train' and r['condition']==c and r['seed']==seed];re=[r for r in records if r['split']=='real' and r['condition']==c and r['seed']==seed]
                def key(r):return (r['point'],r['error_bin'])+(() if axis=='none' else (r[axis],))
                groups={}
                for r in tr:groups.setdefault(key(r),[]).append(r)
                supported=[r for r in re if key(r) in groups]
                if not supported:continue
                out=dict(condition=c,axis=axis,seed=seed,total_real=len(re),supported_real=len(supported),real={},matched_train={})
                for metric in ('base_px','full_px','wrong','alpha'):
                    out['real'][metric]=float(np.mean([r[metric] for r in supported]));out['matched_train'][metric]=float(np.mean([np.mean([t[metric] for t in groups[key(r)]]) for r in supported]))
                matched.append(out)
    write(RUN/'DIAGNOSIS.json',dict(complete=True,rows=table,matched=matched,source_bindings=bindings,scope='V23 new512/new128 only plus real297. Prior253 omitted here; all retained for subsequent training. Projected box aspect is not camera yaw. Predicted box retention is not certified edge visibility. Point-ID/error-bin matching and real GT are posthoc diagnosis only. Conditioning changes common support; report counts.'))
    with (RUN/'DIAGNOSIS_POINTS.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    record('1단계 완료: V23 신규512/확인128 및 실사297의 세 시드, 약한/큰 잘림을 잘리는 쪽·정규화 화면 밖 거리·박스 투영비·잔존율로 분해. DIAGNOSIS.json에 공통 지원과 점ID/오차 구간 통제를 기록. 실제 가시 경계 정답 또는 단독 원인 증명은 아님.')
    for r in table:
        if r['condition']=='crop20' and r['axis'] in ('all','side'):print(r,flush=True)
    for r in matched:
        if r['condition']=='crop20':print('MATCH',r,flush=True)

if __name__=='__main__':main()
