"""Audit corrected calibration and distinguish unapplied proposals from output."""
import re
import subprocess
import sys
import cv2
import numpy as np
import torch
from . import dino_joint_translate_exact as X
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

T=X.T;C=X.C;D=X.D;N=T.N;P=T.P;W=T.W;V=T.V;M=T.M


@torch.no_grad()
def main():
    with X.scope():p=T.verify()
    N.setup();print('GPU',N.E.gpu(),flush=True)
    tests=[C.HERE/n for n in ['test_dino_joint_translate_exact.py','test_dino_joint_translate_model.py','test_dino_multipeak_model.py','test_dino_source_diversity.py','test_dino_visual_long.py','test_dino_wide_visual_model.py','test_dino_wide_model.py','test_dino_localization_model.py','test_heatmap_mode_decoder.py','test_recovery_pseudo_denoise.py']]
    run=subprocess.run([sys.executable,'-m','pytest','-q',*map(str,tests)],capture_output=True,text=True);print(run.stdout,flush=True)
    assert run.returncode==0,run.stderr;ntests=int(re.search(r'(\d+) passed',run.stdout).group(1));assert ntests==42
    parent_audit=C.read(T.DOC/'COMPLETION_AUDIT.json')
    for b in parent_audit['evidence']:C.verify(b)
    assert C.read(X.RAW/'SOURCE_PROPOSALS.json')==C.read(T.RAW/'SOURCE_PROPOSALS.json')
    rows=C.read(X.RAW/'SOURCE_PROPOSALS.json')['records'];cal=[r for r in rows if r['row'] in p['calibration_rows']]
    selected,grid=X.calibrate(cal);fit=C.read(X.DOC/'FIT_REGISTER.json')
    assert selected==fit['calibration'] and grid==fit['grid'] and selected['threshold']==fit['threshold']
    assert selected['threshold']==C.read(T.DOC/'EXACT_SOURCE_FRONTIER_DIAGNOSTIC.json')['best']['threshold']
    test=[r for r in rows if r['row'] in p['test_rows']]
    assert fit['test_raw']==T.stats([r for r in test if r['view']==0],fit['threshold']) and fit['test_combined']==T.stats(test,fit['threshold'])
    for b in C.read(X.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    C.verify(C.read(X.DOC/'DECISION_LOCK.json')['fit']);models=T.heads();backbone,_=D.A.load()
    originals={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    base={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    outputs={r['id']:r for r in C.read(X.RAW/f'EVAL_PREDICTIONS_{T.ARM}.json')['records']}
    receipts={r['id']:r for r in C.read(X.RAW/'INFERENCE_RECEIPTS.json')}
    assert set(originals)==set(outputs)==set(receipts) and len(outputs)==194
    unapplied={};gains=[];modified_candidates=0
    for i,r in enumerate(C.read(X.DOC/'EVAL_PROTOCOL.json')['records']):
        C.verify(r['image']);key=r['id'];old=originals[key];im=cv2.imread(str(C.ROOT/r['image']['path']));x=W.prepare_input(im,old['prediction'])
        feat=W.extract(backbone,[x])[0];assert N.array_sha(feat)==receipts[key]['feature_sha']
        t=torch.as_tensor(feat,device='cuda').float()[None];q=torch.as_tensor(x['points'],device='cuda')[None];v=torch.as_tensor(x['valid'],device='cuda')[None]
        z={a:m(t,q,v) for a,m in models.items()};proposal=M.propose(M.logmass(z['SYN'],z['MIX']),q[0],v[0])
        assert proposal==receipts[key]['proposal'];gains.append(proposal['gain']);modified_candidates+=int(M.changed(proposal))
        apply=M.changed(proposal) and proposal['gain']>fit['threshold'];assert apply==receipts[key]['applied']
        new=M.restore(x['original_points'],proposal,x['matrix'],apply);pred=outputs[key]['prediction']
        np.testing.assert_array_equal(new,np.asarray(P.top(pred)['keypoints_xy']));P.assert_preserved(old['prediction'],pred)
        raw=M.restore(x['original_points'],proposal,x['matrix']);unapplied[key]=raw
        if proposal['available']:
            perm=np.asarray(M.PERMS[proposal['permutation']]);delta=np.asarray(proposal['delta_crop'])@np.linalg.inv(x['matrix'])[:2,:2].T
            np.testing.assert_allclose(raw[:8]-x['original_points'][perm[:8]],np.tile(delta,(8,1)),atol=1e-10,rtol=0)
        if not apply:assert pred==old['prediction']
        if (i+1)%50==0:print('REGISTER_EXACT_REPRODUCE',i+1,N.E.gpu(),flush=True)
    # All candidate proposals were locked before this GT-only diagnosis.
    pe,pop=D.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in outputs}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    metrics=[];corners=[];before=[]
    for key,q in unapplied.items():
        b=base[key];target=targets[key];gt=np.asarray(target.keypoints_xy)
        actual=EM.measure(q,gt,target.keypoint_supervision_mask,perms,outputs[key]['raw_hw'],b['matched'],b['detected'])
        metrics.append(dict(b,**actual));before.append(b)
        if not b['matched']:continue
        oldq=np.asarray(P.top(originals[key]['prediction'])['keypoints_xy'])[:8];valid=np.isfinite(oldq).all(-1)&~(oldq==-1).all(-1)
        for j,(a,c) in enumerate(zip(b['canonical_errors'],actual['canonical_errors'])):
            if a is None:continue
            corners.append(dict(id=key,GT_corner=j,before=a,unapplied_after=c,nearest_R0_point_distance=float(np.linalg.norm(oldq[valid]-gt[j],axis=-1).min()),
                proposed_gain=receipts[key]['proposal']['gain'],actually_applied=receipts[key]['applied']))
    recovery=P.recovery_damage(before,metrics,True);far=[r for r in corners if r['nearest_R0_point_distance']>40 and r['before']>20];assert len(far)==64
    example='eval_pallet07:1778652166837872128'
    C.freeze(X.DOC/'UNAPPLIED_PROPOSAL_DIAGNOSTIC.json',dict(status='NOT_ACTUAL_OUTPUT_NOT_NEW_THRESHOLD',
        whole194_proposals_modified=modified_candidates,gain=dict(minimum=float(min(gains)),median=float(np.median(gains)),maximum=float(max(gains))),
        fixed_source_threshold=fit['threshold'],unapplied_recovery=recovery,
        far_spatial64_unapplied_recovered=sum(r['unapplied_after']<=10 for r in far),rows=corners,
        example=dict(R0=base[example],unapplied=next(r for r in metrics if r['id']==example),receipt=receipts[example]),
        actual_predictions_changed_by_this_diagnostic=False,warning='Removingthelockedgatewouldbea different,unapproved method. GTdiagnosticscannotjustifyrealGTthresholdtuning.'))
    result=C.read(X.DOC/'RESULTS.json');assert result['applied_frames']==sum(r['applied'] for r in receipts.values())
    C.freeze(X.DOC/'COMPLETION_AUDIT.json',dict(experiment_complete=True,goal_complete=False,tests_passed=ntests,
        source_proposals_reused_exactly=320,source_CPU_replay_parent=C.bound(T.DOC/'COMPLETION_AUDIT.json'),
        calibration_full_frontier_reproduced=True,threshold=fit['threshold'],GPU_real_proposals_reproduced=194,
        final_predictions_reproduced=194,max_reproduction_px=0,relative_shape_preserved=True,
        actual_changed_frames=result['applied_frames'],new_annotations=0,new_tags=0,auto_promoted=False,
        evidence=[C.bound(__file__)]+[C.bound(t) for t in tests]+[C.bound(X.DOC/f) for f in ['PROTOCOL.json','FIT_REGISTER.json','DECISION_LOCK.json','OUTPUTS_LOCK.json','RESULTS.json','UNAPPLIED_PROPOSAL_DIAGNOSTIC.json']]))
    print('REGISTER_EXACT_AUDIT_COMPLETE',dict(actual_changes=result['applied_frames'],gains_max=max(gains),threshold=fit['threshold'],unapplied_recovery=recovery,unapplied_far_recovered=sum(r['unapplied_after']<=10 for r in far)),flush=True)


if __name__=='__main__':main()
