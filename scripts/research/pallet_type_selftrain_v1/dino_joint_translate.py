"""Source-calibrated whole-pattern registration; no new frame rejection."""
import argparse
import copy
import cv2
import numpy as np
import torch
from . import dino_multipeak as Q
from . import dino_joint_translate_model as M
from . import dino_joint_model as J

C=Q.C;X=Q.X;D=Q.D;N=Q.N;P=Q.P;W=Q.W;V=Q.V
PHASE='dino_joint_translate';DOC=D.R.DOC/PHASE;RAW=D.R.RAW/PHASE;ARM='REGISTER'
SIDECAR=C.ROOT/'data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz'


def prepare():
    X.verify();split=C.read(D.R.DOC/'dino_joint/PROTOCOL.json')
    cal=split['calibration_rows'];test=split['test_rows'];assert len(cal)==len(test)==32 and not set(cal)&set(test)
    assert set(cal+test)==set(C.read(X.DOC/'PROTOCOL.json')['source_held_rows'])
    p=dict(arms=[ARM],calibration_rows=cal,test_rows=test,
        objective='Test joint learned image evidence for coherent large displacement,without splicing incompatible per-corner proposals.',
        models='Frozen source-diversity SYN/MIX image-only heads and backbone;no new neural training.',
        proposal='Average the two spatial softmax probabilities;5x5 local mass floor1e-12. Score mean logmass at all8corners. Search one shared translation dx=-288..288,dy=-384..384crop_px atstride4 and4 whole graph-preserving index permutations. All8candidatepoints must lie in heatmap domain. No newframefilter;incomplete input is unchanged fallback.',
        roles='Four whole index permutations are actual edits,not extra scoring symmetries. Native center retained. Original relative8pointshape retained under permutation/translation;doesnotguarantee correctcuboid if R0shape is wrong.',
        source='Same64source held,existing32calibration/32test split. Five views:raw,two fixedrandom whole translations,wholequarterpermutation,quarter+translation. Corrupt raw input points,not GT. Allimagefeatures/targets fixed.',
        views='SeedSequence20261007,row,view. View1 displacement .1-.3boxdiag,view2 .3-.6,view4 .1-.6;angleuniform;view3quarteronly. Center unchanged. No realGT-dependent direction.',
        calibration='Gainthreshold0..16 by.25 +1e9null. Selected-case beneficialprecision>=95% over raw+artificial calibration views;rawcalibration PCK10drop<=1pp,PCK20>=R0,P90<=1.1R0,gooddamage<=1%. Maxrawrecoveries,thencombinedrecoveries,thenhigherthreshold. Fixedtest32,no retune fromreal.',
        limitations='Repeatedsource/realDEV;MIXteacherhistoricalmanual3overlap. Wholetranslation cannotrepairdeformedrelativegeometry. Candidatepattern search is not PnP/CAD/depth or geometricframefilter. No auto promotion.',
        new_annotations=0,new_tags=0,new_training_steps=0,auto_promote=False,
        sources=[C.bound(f) for f in [__file__,M.__file__,C.HERE/'test_dino_joint_translate_model.py',J.__file__,SIDECAR,
            X.DOC/'PROTOCOL.json',X.DOC/'CACHE_COMPLETE.json',X.DOC/'COMPLETION_AUDIT.json',X.RAW/'INFERENCE_RECEIPTS.json',
            Q.DOC/'RESULTS.json',Q.DOC/'LIMITATIONS_DIAGNOSTIC.json',Q.DOC/'COMPLETION_AUDIT.json',D.R.DOC/'dino_joint/PROTOCOL.json']]+[C.bound(X.DOC/f'FIT_{a}.json') for a in D.ARMS])
    C.freeze(DOC/'PROTOCOL.json',p);D.R.evaluation_protocol(PHASE,[ARM],p['sources']+[C.bound(DOC/'PROTOCOL.json')])
    print('REGISTER_PROTOCOL_LOCKED',flush=True)


def verify():
    p=C.read(DOC/'PROTOCOL.json')
    for b in p['sources']:C.verify(b)
    return p


def heads():
    out={}
    for a in D.ARMS:
        f=C.read(X.DOC/f'FIT_{a}.json');C.verify(f['checkpoint']);m=V.V.Head().cuda()
        m.load_state_dict(torch.load(C.ROOT/f['checkpoint']['path'],map_location='cpu',weights_only=False)['model']);out[a]=m.eval()
    return out


def view(r,j):
    q=r['points'].copy();valid=r['valid'].copy()
    if j in [3,4]:q=q[np.asarray(M.PERMS[2])].copy();valid=valid[np.asarray(M.PERMS[2])].copy()
    if j in [1,2,4]:
        rng=np.random.default_rng(np.random.SeedSequence([20261007,r['row'],j]));angle=rng.uniform(0,2*np.pi)
        lo,hi={1:(.1,.3),2:(.3,.6),4:(.1,.6)}[j];radius=rng.uniform(lo,hi)*r['bbox_diagonal']*r['matrix'][0,0]
        q[:8]+=radius*np.array([np.cos(angle),np.sin(angle)])
    q[8]=r['points'][8];return q,valid


@torch.no_grad()
def source_cache():
    p=verify();N.setup();print('GPU',N.E.gpu(),flush=True);models=heads();side=np.load(SIDECAR);src=P.SourceData()
    np.testing.assert_array_equal(side['record_index'],src.data.indices)
    cache={r.get('row'):r for r in C.read(X.DOC/'CACHE_COMPLETE.json')['records'] if r['domain']=='source'};records=[]
    for i,row in enumerate(p['calibration_rows']+p['test_rows']):
        meta=cache[row];C.verify(meta['cache'])
        with np.load(C.ROOT/meta['cache']['path']) as z:r=dict(meta,**{k:np.array(z[k]) for k in z.files if k!='protocol_sha256'})
        b=D.tensor_batch([r]);zz={a:m(b['feature'],b['points'],b['valid']) for a,m in models.items()};lp=M.logmass(zz['SYN'],zz['MIX'])
        perms=side['permutations'][row,:int(side['order'][row])]
        for j in range(5):
            q,v=view(r,j);proposal=M.propose(lp,torch.as_tensor(q,device='cuda'),torch.as_tensor(v,device='cuda'))
            after=M.restore(q,proposal,np.eye(3));err=J.canonical_errors(np.stack([q,after]),r['target'],r['target_valid'],perms,r['matrix'][0,0])
            records.append(dict(row=row,view=j,proposal=proposal,errors=[[None if not np.isfinite(v) else float(v) for v in line] for line in err]))
        if (i+1)%16==0:print('REGISTER_SOURCE',i+1,N.E.gpu(),flush=True)
    C.freeze(RAW/'SOURCE_PROPOSALS.json',dict(records=records,GT_used_only_for_diagnostic_labels=True,inputs_include_no_GT=True))


def stats(rows,threshold):
    before=np.asarray([r['errors'][0] for r in rows],float);after=np.asarray([r['errors'][1] for r in rows],float)
    chosen=np.array([M.changed(r['proposal']) and r['proposal']['gain']>threshold for r in rows]);out=np.where(chosen[:,None],after,before)
    valid=np.isfinite(before);b=before[valid];q=out[valid]
    beneficial=(np.nanmean(after,axis=-1)+5<np.nanmean(before,axis=-1))&~((before<5)&(after>10)).any(-1)
    return dict(n=len(b),PCK10=float((q<=10).mean()),PCK20=float((q<=20).mean()),P90=float(np.quantile(q,.9)),
        input_PCK10=float((b<=10).mean()),input_PCK20=float((b<=20).mean()),input_P90=float(np.quantile(b,.9)),
        hard=int((b>20).sum()),recovered=int(((b>20)&(q<=10)).sum()),good=int((b<5).sum()),damaged=int(((b<5)&(q>10)).sum()),
        selected=int(chosen.sum()),beneficial_precision=float(beneficial[chosen].mean()) if chosen.any() else 1.)


def fit():
    p=verify();rows=C.read(RAW/'SOURCE_PROPOSALS.json')['records'];cal=[r for r in rows if r['row'] in p['calibration_rows']]
    raw=[r for r in cal if r['view']==0];grid=[];best=None
    for t in [*np.arange(0,16.001,.25).tolist(),1e9]:
        s=stats(raw,t);allstats=stats(cal,t)
        feasible=s['PCK10']>=s['input_PCK10']-.01 and s['PCK20']>=s['input_PCK20'] and s['P90']<=1.1*s['input_P90'] and s['damaged']<=.01*s['good'] and allstats['beneficial_precision']>=.95
        r=dict(threshold=t,feasible=feasible,raw=s,combined=allstats);grid.append(r);key=(s['recovered'],allstats['recovered'],t)
        if feasible and (best is None or key>best[0]):best=(key,r)
    assert best is not None;selected=best[1];threshold=selected['threshold'];test=[r for r in rows if r['row'] in p['test_rows']]
    C.freeze(DOC/'FIT_REGISTER.json',dict(threshold=threshold,calibration=selected,grid=grid,
        test_raw=stats([r for r in test if r['view']==0],threshold),test_combined=stats(test,threshold),
        source_proposals=C.bound(RAW/'SOURCE_PROPOSALS.json'),GT_real_used=False,new_training_steps=0))
    C.freeze(DOC/'DECISION_LOCK.json',dict(fit=C.bound(DOC/'FIT_REGISTER.json'),before_real_inference=True))
    print('REGISTER_FIT',selected,'test',stats([r for r in test if r['view']==0],threshold),flush=True)


@torch.no_grad()
def infer():
    verify();N.setup();print('GPU',N.E.gpu(),flush=True);models=heads();backbone,_=D.A.load()
    threshold=C.read(DOC/'FIT_REGISTER.json')['threshold'];C.verify(C.read(DOC/'DECISION_LOCK.json')['fit'])
    original={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']};outputs=[];receipts=[]
    oldfeatures={r['id']:r['feature_sha'] for r in C.read(X.RAW/'INFERENCE_RECEIPTS.json')}
    for i,r in enumerate(C.read(DOC/'EVAL_PROTOCOL.json')['records']):
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));old=original[r['id']];x=W.prepare_input(im,old['prediction'])
        feat=W.extract(backbone,[x])[0];assert N.array_sha(feat)==oldfeatures[r['id']]
        t=torch.as_tensor(feat,device='cuda').float()[None];q=torch.as_tensor(x['points'],device='cuda')[None];v=torch.as_tensor(x['valid'],device='cuda')[None]
        z={a:m(t,q,v) for a,m in models.items()};proposal=M.propose(M.logmass(z['SYN'],z['MIX']),q[0],v[0]);apply=M.changed(proposal) and proposal['gain']>threshold
        new=M.restore(x['original_points'],proposal,x['matrix'],apply);pred=copy.deepcopy(old['prediction']);P.top(pred)['keypoints_xy']=new.tolist();P.assert_preserved(old['prediction'],pred)
        outputs.append(dict(id=r['id'],kind='PLASTIC',raw_hw=old['raw_hw'],prediction=pred))
        receipts.append(dict(id=r['id'],proposal=proposal,applied=apply,matrix=x['matrix'].tolist(),feature_sha=N.array_sha(feat)))
        if (i+1)%50==0:print('REGISTER_INFER',i+1,N.E.gpu(),flush=True)
    C.freeze(RAW/f'EVAL_PREDICTIONS_{ARM}.json',dict(complete=True,arm=ARM,records=outputs,GT_free=True,
        checkpoint=C.bound(DOC/'DECISION_LOCK.json'),checkpoint_type='frozen_heads_and_calibration_manifest',image_fits={a:C.bound(X.DOC/f'FIT_{a}.json') for a in D.ARMS}))
    C.freeze(RAW/'INFERENCE_RECEIPTS.json',receipts)
    C.freeze(DOC/'OUTPUTS_LOCK.json',dict(artifacts=[C.bound(RAW/f'EVAL_PREDICTIONS_{ARM}.json'),C.bound(RAW/'INFERENCE_RECEIPTS.json')],before_GT_scoring=True))


def report():
    verify()
    for b in C.read(DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    score=D.R.score(PHASE,ARM,False)['metrics'];base=[r for r in C.read(D.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC']
    scores={r['id']:r for r in score};score=[scores[r['id']] for r in base]
    summary=P.summary(score);rec=P.recovery_damage(base,score,True);original_summary=P.summary(base)
    source=C.read(DOC/'FIT_REGISTER.json')['test_raw']
    frames=sum(b['matched'] and any(x is not None and x>20 and y<=10 for x,y in zip(b['canonical_errors'],n['canonical_errors'])) for b,n in zip(base,score))
    checks=dict(recovery5=rec['recovered']>=5,frames3=frames>=3,damage1percent=rec['damage_rate']<=.01,
        pck20_preserved=summary['PCK']['20']>=original_summary['PCK']['20'],source_PCK10=source['PCK10']>=source['input_PCK10']-.01,
        source_P90=source['P90']<=1.1*source['input_P90'])
    receipts=C.read(RAW/'INFERENCE_RECEIPTS.json');example='eval_pallet07:1778652166837872128'
    C.freeze(DOC/'RESULTS.json',dict(summary=summary,R0_summary=original_summary,recovery=rec,recovered_frames=frames,checks=checks,passed=all(checks.values()),
        applied_frames=sum(r['applied'] for r in receipts),example=dict(R0=next(r for r in base if r['id']==example),REGISTER=scores[example],receipt=next(r for r in receipts if r['id']==example)),
        goal_complete=False,auto_promoted=False,independent_confirmation=False))
    print('REGISTER_RESULT',summary,rec,'frames',frames,'checks',checks,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);a=parser.parse_args()
    if a.action=='prepare':prepare()
    else:source_cache();fit();infer();report()
