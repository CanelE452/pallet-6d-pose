"""Fixed top5 local-mode support diagnostic, never GT-selected inference."""
import argparse
import numpy as np
import cv2
import torch
from . import dino_source_diversity as X
from . import dino_multipeak_model as M

C=X.C;D=X.D;N=X.N;P=X.P;W=X.W;V=X.V
PHASE='dino_multipeak';DOC=D.R.DOC/PHASE;RAW=D.R.RAW/PHASE


def prepare():
    X.verify()
    for b in C.read(X.DOC/'COMPLETION_AUDIT.json')['evidence']:C.verify(b)
    C.freeze(DOC/'PROTOCOL.json',dict(
        objective='Distinguish absent native point coordinates from alternate spatial peaks that the fixed argmax decoder discards.',
        models='Frozen dino_source_diversity SYN/MIX5000;no training or prediction replacement.',
        decoder='Top5 local maxima of each heatmap,5x5 maxpool;greedyChebyshev suppression radius8grid=32crop_px;original5x5local expectation around each peak. Rank by original peak logit,not GT. Rank1 must equal original decoder exactly. Missing modes flagged invalid.',
        prespecified_counts=[1,2,3,5],separation_grid_px=8,
        population='All194PLASTIC plus unchanged64source held. Candidate sets generated and locked before loading GT for diagnostic scoring.',
        scores='Far64realGTcorners:unionR0+allnativechannels from each/combined heads atK=1/2/3/5;minimumdistance andwithin10/20. This ignoresrole/globalpose consistency and is an optimistic oracle,NOT actual model performance. No chosen point applied.',
        source_scores='Same510heldsourcecorners:within10 support from samechannel,officialsymmetry-channel union andallchannels. Per-corner symmetry union is not a single coherent whole-pose choice.',
        no_new_annotations=True,no_new_tags=True,no_new_filters=True,auto_promote=False,
        sources=[C.bound(f) for f in [__file__,M.__file__,C.HERE/'test_dino_multipeak_model.py',X.DOC/'PROTOCOL.json',X.DOC/'COMPLETION_AUDIT.json',X.DOC/'CACHE_COMPLETE.json',X.DOC/'EVAL_PROTOCOL.json',X.DOC/'SPATIAL_SUPPORT_DIAGNOSTIC.json',
            X.RAW/'INFERENCE_RECEIPTS.json',D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json',
            C.ROOT/'data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz']]+[C.bound(X.DOC/f'FIT_{a}.json') for a in D.ARMS]))
    print('MULTIPEAK_PROTOCOL_LOCKED',flush=True)


def verify():
    p=C.read(DOC/'PROTOCOL.json')
    for b in p['sources']:C.verify(b)
    return p


@torch.no_grad()
def infer():
    verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    models={}
    for a in D.ARMS:
        f=C.read(X.DOC/f'FIT_{a}.json');C.verify(f['checkpoint']);ck=torch.load(C.ROOT/f['checkpoint']['path'],map_location='cpu',weights_only=False)
        m=V.V.Head().cuda();m.load_state_dict(ck['model']);models[a]=m.eval()
    backbone,_=D.A.load();ev=C.read(X.DOC/'EVAL_PROTOCOL.json')['records']
    originals={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']}
    oldpred={a:{r['id']:r for r in C.read(X.RAW/f'EVAL_PREDICTIONS_{a}.json')['records']} for a in D.ARMS}
    receipts={r['id']:r for r in C.read(X.RAW/'INFERENCE_RECEIPTS.json')}
    real={a:[] for a in D.ARMS};ids=[]
    for i,r in enumerate(ev):
        C.verify(r['image']);old=originals[r['id']];im=cv2.imread(str(C.ROOT/r['image']['path']));x=W.prepare_input(im,old['prediction'])
        feat=W.extract(backbone,[x])[0];assert N.array_sha(feat)==receipts[r['id']]['feature_sha']
        t=torch.as_tensor(feat,device='cuda').float()[None];q=torch.as_tensor(x['points'],device='cuda')[None];v=torch.as_tensor(x['valid'],device='cuda')[None]
        for a,m in models.items():
            z=m(t,q,v);modes=M.modes(z);torch.testing.assert_close(modes['points'][:,:,0],V.V.decode(z),atol=0,rtol=0)
            out={k:val[0].cpu().numpy() for k,val in modes.items()};out['points']=N.C.transform_points(out['points'].reshape(-1,2),np.linalg.inv(x['matrix'])).reshape(9,5,2)
            out['points'][~x['valid']]=x['original_points'][~x['valid'],None,:];out['points'][8]=x['original_points'][8]
            out['valid']&=x['valid'][:,None];out['valid'][8]=False
            np.testing.assert_array_equal(out['points'][:,0],np.asarray(P.top(oldpred[a][r['id']]['prediction'])['keypoints_xy']))
            real[a].append(out)
        ids.append(r['id'])
        if (i+1)%50==0:print('MULTIPEAK_REAL',i+1,N.E.gpu(),flush=True)
    # Only source input features/points/valid are passed to heads, not targets.
    held=C.read(X.DOC/'PROTOCOL.json')['source_held_rows'];cache={r.get('row'):r for r in C.read(X.DOC/'CACHE_COMPLETE.json')['records'] if r['domain']=='source'}
    source={a:[] for a in D.ARMS}
    for row in held:
        r=cache[row];C.verify(r['cache'])
        with np.load(C.ROOT/r['cache']['path']) as f:
            t=torch.as_tensor(np.array(f['feature']),device='cuda').float()[None];q=torch.as_tensor(np.array(f['points']),device='cuda')[None];v=torch.as_tensor(np.array(f['valid']),device='cuda')[None]
        for a,m in models.items():
            z=m(t,q,v);out=M.modes(z);torch.testing.assert_close(out['points'][:,:,0],V.V.decode(z),atol=0,rtol=0)
            source[a].append({k:val[0].cpu().numpy() for k,val in out.items()})
    bindings=[]
    for name,rows,keys in [('real',real,ids),('source',source,held)]:
        path=RAW/f'{name.upper()}_PROPOSALS.npz';path.parent.mkdir(parents=True,exist_ok=True)
        arrays={a+'_'+k:np.stack([r[k] for r in rr]) for a,rr in rows.items() for k in rr[0]}
        with path.open('xb') as f:np.savez(f,ids=np.asarray(keys),**arrays)
        bindings.append(C.bound(path))
    C.freeze(DOC/'PROPOSALS_LOCK.json',dict(artifacts=bindings,protocol=C.bound(DOC/'PROTOCOL.json'),
        GT_free=True,before_GT_scoring=True,top1_exact_reproductions=516,new_training_steps=0))
    print('MULTIPEAK_PROPOSALS_LOCKED',flush=True)


def score():
    verify()
    for b in C.read(DOC/'PROPOSALS_LOCK.json')['artifacts']:C.verify(b)
    with np.load(RAW/'REAL_PROPOSALS.npz') as z:real={k:np.array(z[k]) for k in z.files}
    with np.load(RAW/'SOURCE_PROPOSALS.npz') as z:source={k:np.array(z[k]) for k in z.files}
    lookup={k:i for i,k in enumerate(real['ids'])};assert len(lookup)==194
    original={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    pe,pop=D.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in lookup}
    far=C.read(X.DOC/'SPATIAL_SUPPORT_DIAGNOSTIC.json')['rows'];assert len(far)==64
    details=[];summary={}
    for name,arms in [('SYN',['SYN']),('MIX',['MIX']),('BOTH',D.ARMS)]:
        summary[name]={}
        for k in [1,2,3,5]:
            distances=[]
            for r in far:
                key=r['id'];j=r['GT_corner'];i=lookup[key];gt=np.asarray(targets[key].keypoints_xy)[j]
                old=np.asarray(P.top(original[key]['prediction'])['keypoints_xy'])[:8];old=old[np.isfinite(old).all(-1)&~(old==-1).all(-1)]
                options=[dict(distance=float(np.linalg.norm(q-gt)),arm='R0',channel=-1,rank=0,mass=None) for q in old]
                for a in arms:
                    for ch in range(8):
                        for rank in range(k):
                            if not real[a+'_valid'][i,ch,rank]:continue
                            q=real[a+'_points'][i,ch,rank];mass=float(real[a+'_mass'][i,ch,rank])
                            options.append(dict(distance=float(np.linalg.norm(q-gt)),arm=a,channel=ch,rank=rank+1,mass=mass))
                best=min(options,key=lambda v:v['distance']);distances.append(best['distance'])
                details.append(dict(id=key,GT_corner=j,pool=name,k=k,nearest_R0_px=r['nearest_R0_px'],oracle_nearest=best))
            a=np.asarray(distances);summary[name][str(k)]=dict(n=64,within10=int((a<=10).sum()),within20=int((a<=20).sum()),median_nearest_px=float(np.median(a)))
    assert summary['BOTH']['1']['within10']==C.read(X.DOC/'SPATIAL_SUPPORT_DIAGNOSTIC.json')['any_point_available_within10']
    # Existing held GT only; no change to stored source metrics/thresholds.
    side=np.load(C.ROOT/'data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz')
    srcdata=P.SourceData();np.testing.assert_array_equal(side['record_index'],srcdata.data.indices)
    cache={r.get('row'):r for r in C.read(X.DOC/'CACHE_COMPLETE.json')['records'] if r['domain']=='source'}
    support={}
    for a in D.ARMS:
        support[a]={}
        for k in [1,2,3,5]:
            counts={key:[] for key in ['same_channel','official_channel_union','any_channel']}
            for i,row in enumerate(source['ids']):
                with np.load(C.ROOT/cache[int(row)]['cache']['path']) as z:gt=np.array(z['target']);mask=np.array(z['target_valid']);gain=float(z['matrix'][0,0])
                perms=side['permutations'][int(row),:int(side['order'][int(row)])]
                for j in np.flatnonzero(mask[:8]):
                    for key,channels in [('same_channel',[j]),('official_channel_union',sorted({int(np.flatnonzero(p==j)[0]) for p in perms})),('any_channel',list(range(8)))]:
                        q=source[a+'_points'][i,channels,:k];valid=source[a+'_valid'][i,channels,:k]
                        distance=np.linalg.norm(q[valid]-gt[j],axis=-1)/gain
                        counts[key].append(float(distance.min()) if len(distance) else float('inf'))
            support[a][str(k)]={key:dict(n=len(v),within10=sum(x<=10 for x in v),within20=sum(x<=20 for x in v)) for key,v in counts.items()}
    C.freeze(DOC/'RESULTS.json',dict(status='GT_ORACLE_CANDIDATE_SUPPORT_NOT_MODEL_PERFORMANCE',real_far64=summary,
        source_held510=support,rows=details,goal_complete=False,actual_predictions_changed=False,
        warning='Role-agnostic pooled modes and per-corner GT choices may not form a coherent pose. Candidate presence is not learned recovery.',
        evidence=[C.bound(DOC/'PROTOCOL.json'),C.bound(DOC/'PROPOSALS_LOCK.json'),C.bound(__file__)]))
    print('MULTIPEAK_REAL_SUPPORT',summary,flush=True);print('MULTIPEAK_SOURCE_SUPPORT',support,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);a=parser.parse_args()
    if a.action=='prepare':prepare()
    else:infer();score()
