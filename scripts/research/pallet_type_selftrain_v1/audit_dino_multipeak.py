"""Reproduce proposals and quantify count/role/weak-mass oracle limitations."""
import re
import subprocess
import sys
from collections import Counter
import cv2
import numpy as np
import torch
from . import dino_multipeak as Q

C=Q.C;X=Q.X;D=Q.D;N=Q.N;P=Q.P;W=Q.W;V=Q.V;M=Q.M


def main():
    Q.verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    tests=[C.HERE/n for n in ['test_dino_multipeak_model.py','test_dino_source_diversity.py','test_dino_visual_long.py','test_dino_wide_visual_model.py','test_dino_wide_model.py','test_dino_localization_model.py','test_heatmap_mode_decoder.py','test_recovery_pseudo_denoise.py']]
    run=subprocess.run([sys.executable,'-m','pytest','-q',*map(str,tests)],capture_output=True,text=True);print(run.stdout,flush=True)
    assert run.returncode==0,run.stderr;ntests=int(re.search(r'(\d+) passed',run.stdout).group(1));assert ntests==33
    for b in C.read(Q.DOC/'PROPOSALS_LOCK.json')['artifacts']:C.verify(b)
    with np.load(Q.RAW/'REAL_PROPOSALS.npz') as z:real={k:np.array(z[k]) for k in z.files}
    with np.load(Q.RAW/'SOURCE_PROPOSALS.npz') as z:source={k:np.array(z[k]) for k in z.files}
    models={}
    for a in D.ARMS:
        f=C.read(X.DOC/f'FIT_{a}.json');C.verify(f['checkpoint']);m=V.V.Head().cuda()
        m.load_state_dict(torch.load(C.ROOT/f['checkpoint']['path'],map_location='cpu',weights_only=False)['model']);models[a]=m.eval()
    originals={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    oldpred={a:{r['id']:r for r in C.read(X.RAW/f'EVAL_PREDICTIONS_{a}.json')['records']} for a in D.ARMS}
    metadata={r['id']:r for r in C.read(X.DOC/'EVAL_PROTOCOL.json')['records']}
    receipts={r['id']:r for r in C.read(X.RAW/'INFERENCE_RECEIPTS.json')}
    assert set(real['ids'])==set(metadata)==set(originals) and len(real['ids'])==194
    backbone,_=D.A.load()
    with torch.no_grad():
        for i,key in enumerate(real['ids']):
            r=metadata[key];C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));x=W.prepare_input(im,originals[key]['prediction'])
            feat=W.extract(backbone,[x])[0];assert N.array_sha(feat)==receipts[key]['feature_sha']
            t=torch.as_tensor(feat,device='cuda').float()[None];q=torch.as_tensor(x['points'],device='cuda')[None];v=torch.as_tensor(x['valid'],device='cuda')[None]
            for a,m in models.items():
                z=m(t,q,v);modes=M.modes(z);torch.testing.assert_close(modes['points'][:,:,0],V.V.decode(z),atol=0,rtol=0)
                out={k:val[0].cpu().numpy() for k,val in modes.items()}
                # Independent local-maximum, non-overlap and window-mass checks.
                logits=z[0].cpu().numpy();lp=logits.astype(float);lp-=lp.max((-2,-1),keepdims=True);prob=np.exp(lp);prob/=prob.sum((-2,-1),keepdims=True)
                for ch in range(8):
                    valid=out['valid'][ch];anchors=out['anchors'][ch][valid]
                    for j,(ax,ay) in enumerate(anchors):
                        patch=logits[ch,max(0,ay-2):ay+3,max(0,ax-2):ax+3]
                        assert logits[ch,ay,ax]==patch.max()
                        mass=prob[ch,max(0,ay-2):ay+3,max(0,ax-2):ax+3].sum()
                        np.testing.assert_allclose(mass,out['mass'][ch,j],atol=2e-6,rtol=2e-6)
                        if j:assert (np.abs(anchors[:j]-[ax,ay]).max(-1)>8).all()
                    assert (np.diff(out['peak_logit'][ch][valid])<=0).all()
                out['points']=N.C.transform_points(out['points'].reshape(-1,2),np.linalg.inv(x['matrix'])).reshape(9,5,2)
                out['points'][~x['valid']]=x['original_points'][~x['valid'],None,:];out['points'][8]=x['original_points'][8]
                out['valid']&=x['valid'][:,None];out['valid'][8]=False
                for k,value in out.items():np.testing.assert_array_equal(value,real[a+'_'+k][i])
                np.testing.assert_array_equal(out['points'][:,0],np.asarray(P.top(oldpred[a][key]['prediction'])['keypoints_xy']))
            if (i+1)%50==0:print('MULTIPEAK_AUDIT_REAL',i+1,N.E.gpu(),flush=True)
        cache={r.get('row'):r for r in C.read(X.DOC/'CACHE_COMPLETE.json')['records'] if r['domain']=='source'}
        for i,row in enumerate(source['ids']):
            r=cache[int(row)];C.verify(r['cache'])
            with np.load(C.ROOT/r['cache']['path']) as f:
                t=torch.as_tensor(np.array(f['feature']),device='cuda').float()[None];q=torch.as_tensor(np.array(f['points']),device='cuda')[None];v=torch.as_tensor(np.array(f['valid']),device='cuda')[None]
            for a,m in models.items():
                z=m(t,q,v);out=M.modes(z);torch.testing.assert_close(out['points'][:,:,0],V.V.decode(z),atol=0,rtol=0)
                for k,value in out.items():np.testing.assert_array_equal(value[0].cpu().numpy(),source[a+'_'+k][i])
    result=C.read(Q.DOC/'RESULTS.json')
    for a in D.ARMS:
        fit=C.read(X.DOC/f'FIT_{a}.json')['source_after'];s=result['source_held510'][a]['1']['same_channel']
        assert s['n']==fit['n'] and s['within10']/s['n']==fit['PCK10'] and s['within20']/s['n']==fit['PCK20']
    # Posthoc chance reference: original R0+top1 retained; replace the other
    # four ranks per head/channel with same-count random coordinates. No tuning.
    pe,pop=D.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in metadata}
    far=C.read(X.DOC/'SPATIAL_SUPPORT_DIAGNOSTIC.json')['rows'];lookup={key:i for i,key in enumerate(real['ids'])}
    rng=np.random.default_rng(20261006);uniform=rng.random((32,len(real['ids']),2,8,4,2));randoms={}
    for domain in ['wide_decoder_rectangle','R0_box']:
        distances=[]
        for r in far:
            key=r['id'];i=lookup[key];j=r['GT_corner'];gt=np.asarray(targets[key].keypoints_xy)[j]
            original=np.asarray(P.top(originals[key]['prediction'])['keypoints_xy'])[:8]
            first=[original]+[real[a+'_points'][i,:8,0][real[a+'_valid'][i,:8,0]] for a in D.ARMS]
            oldbest=float(np.linalg.norm(np.concatenate(first)-gt,axis=-1).min())
            if domain=='wide_decoder_rectangle':
                q=uniform[:,i]*[572,764];matrix=np.linalg.inv(np.asarray(receipts[key]['matrix']))
                q=N.C.transform_points(q.reshape(-1,2),matrix).reshape(32,2,8,4,2)
            else:
                box=np.asarray(P.top(originals[key]['prediction'])['box_xyxy']);q=box[:2]+uniform[:,i]*(box[2:]-box[:2])
            valid=np.stack([real[a+'_valid'][i,:8,1:] for a in D.ARMS]);d=np.linalg.norm(q-gt,axis=-1)
            d=np.where(valid[None],d,np.inf).reshape(32,-1).min(-1);distances.append(np.minimum(d,oldbest))
        counts=(np.asarray(distances)<=10).sum(0)
        randoms[domain]=dict(replicates=32,seed=20261006,same_added_candidate_counts=True,
            within10_counts=counts.tolist(),mean=float(counts.mean()),minimum=int(counts.min()),maximum=int(counts.max()),
            warning='Chance reference only,not a calibrated statistical test or coherent pose inference.')
    one={(r['id'],r['GT_corner']):r for r in result['rows'] if r['pool']=='BOTH' and r['k']==1}
    added=[r for r in result['rows'] if r['pool']=='BOTH' and r['k']==5 and r['oracle_nearest']['distance']<=10 and one[(r['id'],r['GT_corner'])]['oracle_nearest']['distance']>10]
    # Correct semantic channel restrictions; still per-corner oracle, not a
    # coherent whole-frame symmetry selection.
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    base={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    roles={}
    for mode in ['R0_frame_branch','either_official_branch']:
        roles[mode]={}
        for k in [1,2,3,5]:
            dd=[]
            for r in far:
                key=r['id'];i=lookup[key];j=r['GT_corner'];gt=np.asarray(targets[key].keypoints_xy)[j]
                allowed=[perms[base[key]['branch']]] if mode=='R0_frame_branch' else perms
                channels=sorted({p.index(j) for p in allowed});qs=[]
                for a in D.ARMS:
                    coords=real[a+'_points'][i,channels,:k];valid=real[a+'_valid'][i,channels,:k];qs.extend(coords[valid])
                dd.append(float(np.linalg.norm(np.asarray(qs)-gt,axis=-1).min()) if qs else float('inf'))
            roles[mode][str(k)]=dict(n=64,within10=sum(d<=10 for d in dd),within20=sum(d<=20 for d in dd))
    masses=[r['oracle_nearest']['mass'] for r in added]
    supplement=dict(status='POSTHOC_GT_DIAGNOSTIC_NOT_METHOD',chance_controls=randoms,role_constrained_support=roles,
        newly_supported=len(added),newly_supported_unique_frames=len({r['id'] for r in added}),
        newly_supported_sessions=dict(Counter(r['id'].split(':')[0] for r in added)),
        nearest_peak_mass=dict(minimum=float(min(masses)),median=float(np.median(masses)),maximum=float(max(masses))),
        newly_supported_rows=added,predictions_changed=False,
        caution='Nearest successful peak is selected using GT for analysis only. Low mass and concentration in one session limit interpretation.')
    C.freeze(Q.DOC/'LIMITATIONS_DIAGNOSTIC.json',supplement)
    C.freeze(Q.DOC/'COMPLETION_AUDIT.json',dict(diagnostic_complete=True,goal_complete=False,tests_passed=ntests,
        real194x2_proposals_reproduced=True,source64x2_proposals_reproduced=True,max_difference=0,
        rank1_real_saved_predictions_exact=388,rank1_source_reference_decoder_exact=128,
        all_valid_real_peaks_local_maxima_separated=True,independent_window_mass_atol=2e-6,
        actual_predictions_changed=False,new_training_steps=0,
        evidence=[C.bound(__file__)]+[C.bound(t) for t in tests]+[C.bound(Q.DOC/f) for f in ['PROTOCOL.json','PROPOSALS_LOCK.json','RESULTS.json','LIMITATIONS_DIAGNOSTIC.json']]))
    print('MULTIPEAK_AUDIT_COMPLETE',dict(chance=randoms,roles=roles,new=len(added),frames=supplement['newly_supported_unique_frames'],sessions=supplement['newly_supported_sessions'],mass=supplement['nearest_peak_mass']),flush=True)


if __name__=='__main__':main()
