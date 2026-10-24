"""Independent saved VAL check: no producer/operator/helper import, no RGB/model IO."""
from pathlib import Path
from collections import Counter
import hashlib,json,math,time
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
S=V/'synthetic_val_preparation';OUT=V/'synthetic_VAL2048_support_v1'
FLAGS=('query_available','factual_query_available','feature_constructed','crop_constructed','context_available','Q_inside','RGB_central4_available','native_continuous_valid','native_nearest_valid')
CLASSES=['OUTSIDE_END_REGION','END_VISIBLE','END_FOREIGN_FRONT','UNKNOWN_SELF_HIDDEN','UNKNOWN_AMBIGUOUS','UNKNOWN_REGISTRATION_OR_CHANGE','INVALID_SUPPORT']
def require(ok,msg):
    if not bool(ok):raise AssertionError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def exact(a,b):return np.array_equal(a,b,equal_nan=True) if np.asarray(a).dtype.kind in 'fc' else np.array_equal(a,b)

def patch_pixels(g):
    # Independently solve the 2x2 Gram system analytically, without operator AST/import.
    K=np.asarray(g['camera']['K'],float);R=np.asarray(g['camera']['R'],float);t=np.asarray(g['camera']['t'],float)
    pc=np.asarray(g['quads_world'],float)@R.T+t;near=g['camera'].get('near',.001);h,w=g['raw_hw']
    if np.any(pc[...,2]<=near):return {},False,True
    hp=pc@K.T;uv=hp[...,:2]/hp[...,2,None]
    oof=bool(np.any(uv[...,0]<0) or np.any(uv[...,0]>w-1) or np.any(uv[...,1]<0) or np.any(uv[...,1]>h-1))
    invK=np.linalg.inv(K);pixels={}
    for face,projection in zip(pc,uv):
        U=face[1]-face[0];W=face[3]-face[0];normal=np.cross(U,W)
        uu=float(U@U);uw=float(U@W);ww=float(W@W);det=uu*ww-uw*uw;require(det>0,'Nondegenerate fixed patch')
        lo=np.maximum(np.floor(projection.min(0)).astype(int),[0,0]);hi=np.minimum(np.ceil(projection.max(0)).astype(int),[w-1,h-1])
        for y in range(int(lo[1]),int(hi[1])+1):
            for x in range(int(lo[0]),int(hi[0])+1):
                ray=invK@np.array([x,y,1.],float);den=float(normal@ray)
                if den==0:continue
                hit=ray*(float(normal@face[0])/den);delta=hit-face[0];du=float(delta@U);dw=float(delta@W)
                a=(du*ww-dw*uw)/det;b=(dw*uu-du*uw)/det
                if hit[2]>near and 0<=a<=1 and 0<=b<=1:
                    pixels[y,x]=min(pixels.get((y,x),math.inf),float(hit[2]))
    return pixels,True,oof

def end_classes(before,scene,patch,primary):
    cls=np.zeros((512,512),np.int8)
    for (y,x),expected in patch.items():
        bp=int(before['part_id'][y,x]);sp=int(scene['part_id'][y,x]);bd=float(before['camera_depth'][y,x]);sd=float(scene['camera_depth'][y,x])
        if before['ambiguity'][y,x] or scene['ambiguity'][y,x]:c=4
        else:
            tol=64*np.finfo(np.float64).eps*max(1.,abs(bd) if math.isfinite(bd) else 0.,abs(expected))
            finite=math.isfinite(bd) and bd>0;target=bp>=0 and bp!=999
            registered=finite and bp==primary and abs(bd-expected)<=tol
            if registered and sp==bp and before['global_triangle_id'][y,x]==scene['global_triangle_id'][y,x] and bd==sd:c=1
            elif registered and sp==999 and math.isfinite(sd) and 0<sd<bd:c=2
            elif target and finite and bd<expected-tol:c=3
            else:c=5
        cls[y,x]=c
    return cls

def positions(q,L,size=96,lo=0,hi=96):
    u=(np.arange(lo,hi,dtype=np.float64)+.5)/size-.5
    x,y=np.meshgrid(q[0]+u*L,q[1]+u*L);return np.column_stack((x.ravel(),y.ravel()))
def mass7(cls,q,L):
    points=positions(q,L);x,y=points.T;xx=np.floor(x).astype(int);yy=np.floor(y).astype(int);fx=x-xx;fy=y-yy
    wt=np.zeros((len(points),7),np.float64)
    for ax,ay in ((0,0),(0,1),(1,0),(1,1)):
        px,py=xx+ax,yy+ay;w=(fx if ax else 1-fx)*(fy if ay else 1-fy);inside=(px>=0)&(px<512)&(py>=0)&(py<512)
        c=np.full(len(points),6,np.int8);c[inside]=cls[py[inside],px[inside]];wt[np.arange(len(points)),c]+=w
    domain=(x>=-.5)&(x<511.5)&(y>=-.5)&(y<511.5);wt[~domain]=0;wt[~domain,6]=1
    return wt.mean(0)
def owner4_scalar(scene,q,L):
    allmass=[]
    for x,y in positions(q,L,lo=47,hi=49):
        if not(-.5<=x<511.5 and -.5<=y<511.5):allmass.append([0.,0.,0.,0.,1.]);continue
        ix=math.floor(x);iy=math.floor(y);fx=x-ix;fy=y-iy;terms=[[] for _ in range(5)]
        for ax,ay,w in ((0,0,(1-fx)*(1-fy)),(0,1,(1-fx)*fy),(1,0,fx*(1-fy)),(1,1,fx*fy)):
            xx,yy=ix+ax,iy+ay
            if not(0<=xx<512 and 0<=yy<512):c=4
            elif scene['ambiguity'][yy,xx]:c=3
            else:
                p=int(scene['part_id'][yy,xx]);c=2 if p<0 else 1 if p==999 else 0
            terms[c].append(w)
        allmass.append([math.fsum(t) for t in terms])
    return np.array([math.fsum(r[k] for r in allmass)/4 for k in range(5)],np.float64)
def load(b,g,used):
    require(sha(b['path'])==b['sha256'],'Original source SHA before');used[b['path']]=b
    keys=['part_id','camera_depth','global_triangle_id','ambiguity','camera_K','camera_R','camera_t','native8_CF_xy','permCF_to_canonical']
    with np.load(b['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in keys}
    for k,dt in [('part_id',np.int32),('camera_depth',np.float64),('global_triangle_id',np.int32),('ambiguity',np.bool_)]:require(a[k].dtype==dt and a[k].shape==(512,512),'Source field schema '+k)
    for k in ('K','R','t'):require(exact(a['camera_'+k],np.asarray(g['camera'][k],np.float64)),'Camera exact '+k)
    require(exact(a['native8_CF_xy'],np.asarray(g['native8_CF_xy'],np.float64)) and exact(a['permCF_to_canonical'],np.asarray(g['permCF_to_canonical'],np.int32)),'Native/CF exact')
    return a

def run():
    start=time.monotonic();p=read(S/'PROTOCOL_BEFORE_VAL_PAYLOADS.json');complete=read(OUT/'COMPLETE.json');require(complete['completed'] and complete['PASS'] and complete['evaluation_only'],'Actual completed VAL barrier')
    require(complete['protocol']==bind(S/'PROTOCOL_BEFORE_VAL_PAYLOADS.json'),'Actual protocol binding')
    old=Path(p['original_operator']['path']).read_text();new=Path(p['operator']['path']).read_text()
    require(new==old.replace("record['split'] == 'TRAIN'","record['split'] in ('TRAIN', 'VALIDATION')"),'Sole operator split assertion delta')
    plan=lines(p['plan']['path']);geo=lines(p['geometry']['path']);saved=lines(complete['rows_output']['path']);gm={(g['base_index'],g['native_slot']):g for g in geo}
    require(len(plan)==len(saved)==2048 and len(gm)==512 and [r['original_global_index'] for r in plan]==list(range(4096,6144)),'All original VAL rows')
    used={};computed=[];maxmass=maxowner=0.;metadata_rows=0
    # Source buffers and independent calculations precede comparison with saved row values.
    for base in range(128,192):
        rr=[r for r in plan if r['base_index']==base];before=load(rr[0]['BEFORE_buffer'],gm[base,0],used);patches={k:patch_pixels(gm[base,k]) for k in range(8)}
        for variant in range(4):
            sr=rr[variant*8:(variant+1)*8];scene=before if variant==0 else load(sr[0]['scene_buffer'],gm[base,0],used)
            for r in sr:
                q,L=r['query_xy'],r['L'];patch,valid,oof=patches[r['native_slot']]
                if q is None or L is None:mass=None;owner=None;v=f=-1
                else:
                    cls=end_classes(before,scene,patch,r['primary_part_id']);mass=mass7(cls,q,L);owner=owner4_scalar(scene,q,L)
                    unknown=float(mass[3]+mass[4]+mass[5]);end=float(mass[1]+mass[2]+unknown)
                    if not r['original_input_flags']['query_available'] or not valid or oof:v=f=-1
                    elif end==0:v=0;f=-1
                    else:v=1 if mass[1]>0 else -1 if unknown>0 else 0;f=1 if mass[2]>0 else -1 if unknown>0 else 0
                computed.append((mass,owner,v,f))
    for i,(r,s,c) in enumerate(zip(plan,saved,computed)):
        for k in ('original_global_index','original_occurrence_id','scene_index','base_index','native_slot','split','variant','query_xy','L','original_input_flags','original_semantic_state'):
            require(s[k]==r[k],'Original metadata '+k)
        require(s['evaluation_only'] and not s['training_admission'] and s['native_visibility_target']==s['point_quality_target']==-1,'Unchanged target authority');metadata_rows+=1
        mass,owner,v,f=c
        if mass is None:require(s['full96_average_mass'] is None and s['owner_central4_soft5'] is None,'Missing null retained')
        else:
            md=float(np.max(np.abs(mass-np.asarray(s['full96_average_mass']))));od=float(np.max(np.abs(owner-np.asarray(s['owner_central4_soft5']))));maxmass=max(maxmass,md);maxowner=max(maxowner,od)
            require(md<=2e-13 and od<=2e-14,'Independent source mass parity')
        require(v==s['visible_support_target'] and f==s['foreign_front_support_target'],'Independent support target parity')
    with np.load(complete['sidecar']['path'],allow_pickle=False) as z:
        require(z['evaluation_only'].item() and not z['training_admission'].item() and z['source_end_class_order'].tolist()==CLASSES,'NPZ authority/classes')
        for k in ('original_global_index','original_occurrence_id','scene_index','base_index','native_slot','split','query_xy','L'):
            vals=np.asarray([[np.nan,np.nan] if r[k] is None else r[k] for r in plan],np.float64) if k=='query_xy' else np.asarray([np.nan if r[k] is None else r[k] for r in plan],np.float64 if k=='L' else None)
            require(exact(z[k],vals),'NPZ original metadata '+k)
        for k in FLAGS:require(exact(z[k],np.asarray([r['original_input_flags'][k] for r in plan],bool)),'NPZ original flag '+k)
        for k,j,n in [('jointmass7',0,7),('owner_central4_soft5',1,5)]:
            vals=np.asarray([[np.nan]*n if c[j] is None else c[j] for c in computed],np.float64);require(np.allclose(z[k],vals,atol=2e-13 if n==7 else 2e-14,rtol=0,equal_nan=True),'NPZ computed '+k)
            js=np.asarray([[np.nan]*n if r['full96_average_mass' if n==7 else 'owner_central4_soft5'] is None else r['full96_average_mass' if n==7 else 'owner_central4_soft5'] for r in saved],np.float64);require(exact(z[k],js),'Saved JSONL/NPZ exact parity '+k)
        for k,j in [('visible_support_target',2),('foreign_front_support_target',3)]:require(exact(z[k],np.asarray([c[j] for c in computed],np.int8)),'NPZ target '+k)
    for b in used.values():require(sha(b['path'])==b['sha256'],'Original source SHA after')
    stats=dict(rows=2048,scenes=256,bases=64,native_counts=dict(Counter(r['native_slot'] for r in plan)),original_input_U=sum(not r['original_input_flags']['query_available'] for r in plan),
        visible_counts=dict(Counter(c[2] for c in computed)),foreign_counts=dict(Counter(c[3] for c in computed)),missing_mass=sum(c[0] is None for c in computed),
        owner5_sum=[math.fsum(float(c[1][j]) for c in computed if c[1] is not None) for j in range(5)])
    require(stats['original_input_U']==8 and len(used)==256,'All VAL/U/source buffers')
    result=dict(completed=True,PASS=True,review_code=bind(__file__),code=p['code'],protocol=bind(S/'PROTOCOL_BEFORE_VAL_PAYLOADS.json'),actual_complete=bind(OUT/'COMPLETE.json'),
        rows_output=complete['rows_output'],sidecar=complete['sidecar'],coverage=stats,metadata_rows_exact=metadata_rows,source_buffer_hashes_before_after=256,
        independent_computation='No producer/operator imports. Scalar ray/plane+analytic2x2 face coordinates for fixed512patches; per-sourcepixel end conjunction; independent bilinear96 aggregation; scalar4sample×4tap math.fsum owner5; then compared saved rows/NPZ.',
        max_abs_jointmass7_difference=maxmass,max_abs_owner5_difference=maxowner,all_support_targets_exact=True,all_JSONL_NPZ_numeric_metadata_and_target_rows_compared=True,
        limitations=['Fixed geometry table/teacher definition is the authority; audit does not certify physical native identity or real-world pose/corner accuracy.','Evaluates previously reused synthetic VAL only. No models, RGB, fit, architecture, selection or thresholds inspected/changed.'],
        evaluation_only=True,new_training_or_inference=0,private_publicheld_V112_reads=0,selected_model=None,fully_solved=False,seconds=time.monotonic()-start)
    f=H/'REVIEW.json';f.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(completed=True,PASS=True,review=bind(f),coverage=stats)))
if __name__=='__main__':run()
