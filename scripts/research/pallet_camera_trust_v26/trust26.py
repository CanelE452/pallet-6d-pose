"""Line accuracy, point accuracy, correspondence and oracle selection separated."""
from context26 import *
from sklearn.metrics import roc_auc_score
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))

def measure(m,ts,r,pred,split,c,seed):
    rows=[]
    for i,(row,t) in enumerate(zip(m['rows'],ts)):
        if not row['detected']:continue
        j=row['observation_index'];b=r['base'][j].numpy().astype(float);lines=r['lines'][j].numpy().astype(float);weights=r['line_weights'][j,:,:-1].numpy();norm=np.linalg.norm(lines[:,:2],axis=-1);line_ok=norm>1e-8;lines=lines/np.maximum(norm[:,None],1e-8);gt=np.array(t['points']);valid=np.array(t['valid']);bb=np.array(row['box']);diag=np.linalg.norm(np.maximum(bb[2:]-bb[:2],1));radius=.15*diag
        eligible=[(ei,a,z) for ei,(a,z) in enumerate(EDGES) if valid[a] and valid[z] and np.linalg.norm(gt[z]-gt[a])>1.]
        if not eligible:continue
        endpoints=np.array([[gt[a],gt[z]] for _,a,z in eligible]);err=np.sqrt(np.mean((np.einsum('lk,epk->lep',lines[:,:2],endpoints)+lines[:,2,None,None])**2,axis=-1));err[~line_ok]=1e12
        for p in np.where(valid)[0]:
            incident=[k for k,(_,a,z) in enumerate(eligible) if p in (a,z)]
            if not incident:continue
            ids=np.argsort(-weights[p],kind='stable')[:3];ids=ids[line_ok[ids]]
            if not len(ids):continue
            first=ids[0];v=gt[p]-b[p];before=float(np.linalg.norm(v));signed=lines[ids,:2]@b[p]+lines[ids,2];delta=-np.clip(signed,-radius,radius)[:,None]*lines[ids,:2];candidate=b[p]+delta;errors=np.linalg.norm(candidate-gt[p],axis=-1)
            nearest_inc=incident[int(err[first,incident].argmin())];nearest_any=int(err[first].argmin());ei,a,z=eligible[nearest_inc];edge=gt[z]-gt[a];tangent=edge/np.linalg.norm(edge);normal=np.array([-tangent[1],tangent[0]]);incident_fit=float(err[first,nearest_inc]);any_fit=float(err[first,nearest_any]);global_incident=p in EDGES[eligible[nearest_any][0]]
            exact=[]
            for ix in incident:
                _,aa,zz=eligible[ix];d=gt[zz]-gt[aa];n=np.array([-d[1],d[0]])/np.linalg.norm(d);sd=float((b[p]-gt[aa])@n);exact.append(b[p]-np.clip(sd,-radius,radius)*n)
            exacterr=np.linalg.norm(np.array(exact)-gt[p],axis=-1);oracle_pred=min(before,float(errors.min()));oracle_exact=min(before,float(exacterr.min()));gt_on_topline=float(abs(lines[first,:2]@gt[p]+lines[first,2]));normal_error=float(abs(v@normal));tangent_error=float(abs(v@tangent));rank2angles=[]
            for ii in range(len(incident)):
                for jj in range(ii):
                    _,aa,zz=eligible[incident[ii]];u=gt[zz]-gt[aa];u/=np.linalg.norm(u);_,aa,zz=eligible[incident[jj]];v0=gt[zz]-gt[aa];v0/=np.linalg.norm(v0);rank2angles.append(abs(float(np.cross(u,v0))))
            out=dict(split=split,condition=c,seed=seed,id=row['id'],session=row['session'],point=int(p),base_px=before,projection_px=float(errors[0]),oracle_predicted_px=oracle_pred,oracle_exact_incident_px=oracle_exact,incident_line_rms_px=incident_fit,any_line_rms_px=any_fit,nearest_role_incident=bool(global_incident),GT_distance_to_predicted_line=gt_on_topline,normal_error_px=normal_error,tangent_error_px=tangent_error,association_score=float(weights[p,first]),line_good=incident_fit<=5,point_good=before<=5,projection_helps=float(errors[0])<before-.1,projection_harms=float(errors[0])>before+.1,correction_normal_dot=float(v@delta[0]),two_incident_independent=bool(rank2angles and max(rank2angles)>.1),null_weight=float(r['line_weights'][j,p,-1]),top3_has_good_incident=bool((err[ids][:,incident]<=5).any()),target_type='outside' if t['outside'][p] else 'hidden' if t.get('hidden',[False]*8)[p] else 'visible')
            out.update(null_preferred=bool(float(r['line_weights'][j,p,-1])>=weights[p,first]),reference_edge_id=int(ei),height_edge=bool(ei in (1,3,5,7)))
            for arm,points in pred.items():out[arm+'_px']=float(np.linalg.norm(points[i,p]-gt[p]))
            if split=='camera':out.update(yaw_deg=row['yaw_deg'],camera_height_m=row['camera_height_m'],source_group=row['source_group'],visible_fraction=row['visible_fraction'])
            rows.append(out)
    return rows

def aggregate(rows):
    keys=('base_px','projection_px','oracle_predicted_px','oracle_exact_incident_px','incident_line_rms_px','normal_error_px','tangent_error_px','line_good','point_good','projection_helps','projection_harms','two_incident_independent','top3_has_good_incident','gate_paired_px','vector_local_edge_px','vector_line_edge_px','null_preferred','height_edge','null_weight','association_score');result=dict(points=len(rows),originals=len({r['id'] for r in rows}))
    for k in keys:
        if rows and k in rows[0]:result[k]=float(np.mean([r[k] for r in rows]))
    if rows:
        y=[r['projection_helps'] for r in rows];result['association_help_auc']=float(roc_auc_score(y,[r['association_score'] for r in rows])) if len(set(y))>1 else None
        result['line_accuracy_help_auc_privileged']=float(roc_auc_score(y,[-r['incident_line_rms_px'] for r in rows])) if len(set(y))>1 else None
    return result

def real_and_synthetic():
    from eval25 import population as pop25
    records=[];coverage=[]
    for split in ('real','prospective'):
        for c in ('clean','corner20','crop20','crop40'):
            m,ts=pop25(split,c)
            for seed in SEEDS:
                r=new_rep('new_check',c,seed) if split=='prospective' else repcache(split,c,seed)
                pred={a:np.load(PREV25.RUN/'predictions'/split/c/f'{a}_seed{seed}.npz')['points'].astype(float) for a in ('gate_paired','vector_local_edge','vector_line_edge')}
                rr=measure(m,ts,r,pred,split,c,seed);records+=rr;coverage.append(dict(split=split,condition=c,seed=seed,all_valid_detected_points=sum(sum(t['valid']) for t,row in zip(ts,m['rows']) if row['detected']),line_reference_supported_points=len(rr),scope='Only points with at least one valid annotated incident segment support line accuracy. Full-model performance keeps all original points.'))
    return records,coverage

def main():
    torch.set_num_threads(2);records,coverage=real_and_synthetic()
    if (RUN/'INFERENCE_COMPLETE.json').exists():
        m=read(RUN/'cache/manifest.json');ts=read(RUN/'TARGETS.json')['rows']
        for seed in SEEDS:
            r=torch.load(RUN/'parents'/f'seed{seed}.pt',weights_only=True,map_location='cpu');pred={a:np.load(RUN/'predictions'/f'{a}_seed{seed}.npz')['points'].astype(float) for a in ('gate_paired','vector_local_edge','vector_line_edge')};rr=measure(m,ts,r,pred,'camera','mixed',seed)
            for x in rr:x['condition']=x['id'].split('::')[-1]
            records+=rr
    table=[];threshold=[]
    for split in sorted({r['split'] for r in records}):
        for c in sorted({r['condition'] for r in records if r['split']==split}):
            for seed in SEEDS:
                pop=[r for r in records if r['split']==split and r['condition']==c and r['seed']==seed]
                for typ in ('all','outside','hidden'):
                    sub=pop if typ=='all' else [r for r in pop if r['target_type']==typ]
                    if not sub:continue
                    table.append(dict(split=split,condition=c,seed=seed,subset=typ,point_quality='all',line_quality='all',**aggregate(sub)))
                    for pg in (False,True):
                        for lg in (False,True):
                            cell=[r for r in sub if r['point_good']==pg and r['line_good']==lg]
                            if cell:table.append(dict(split=split,condition=c,seed=seed,subset=typ,point_quality='good' if pg else 'bad',line_quality='good' if lg else 'bad',**aggregate(cell)))
                    for tol in (2,5,10):
                        good=[r for r in sub if r['incident_line_rms_px']<=tol];wrongrole=[r for r in sub if r['any_line_rms_px']<=tol and r['incident_line_rms_px']>tol]
                        threshold.append(dict(split=split,condition=c,seed=seed,subset=typ,tolerance_px=tol,compatible_points=len(good),compatible_projection_harm_fraction=float(np.mean([r['projection_harms'] for r in good])) if good else None,good_geometry_wrong_role_points=len(wrongrole)))
    with (RUN/'TRUST_POINTS.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
    write(RUN/'TRUST_SUMMARY.json',dict(complete=True,rows=table,GT_tolerance_sensitivity=threshold,coverage=coverage,
        definitions='Line compatibility = RMS signed distances of BOTH annotated incident-edge endpoints to top-associated NON-NULL DHT infinite line <=5px;2/10 sensitivity. This is camera-facing cuboid annotation geometry, not true pixel boundary precision or visibility. Predicted projection forcibly uses top non-null line, displacement bounded15%boxdiag; existing model output and null preference separately recorded. This forced projection is not the actual learned inference rule. oracle_predicted = GT min(base,top3projected candidates), oracle_exact = GT min(base,exact incident-line projections). Neither oracle is a deployed prediction.',
        limits='Line-reference subset may omit missing manual endpoints. Correlated points/seed repetitions; per-seed tables used, not enlarged independent sample size. GT-aware quality strata and exact-line oracle are privileged diagnostics. No fitted routing/threshold or real GT training. New renderer is a controlled probe, not an identical source-domain replay.'))
    for r in table:
        if r['split']=='real' and r['condition']=='crop20' and r['seed']==1 and r['subset']=='outside':print('TRUST26',r,flush=True)
    record('점/선 신뢰도 분해 완료: 선 역할 대응·수직/접선 잔차·좋은점/좋은선4분면·association 점수와 보정 이득·GT 최적 후보 상한을 분리 기록. Oracle를 학습 또는 추론 규칙으로 사용하지 않음.')

if __name__=='__main__':main()
