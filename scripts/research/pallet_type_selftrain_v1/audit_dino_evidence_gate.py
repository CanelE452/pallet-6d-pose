"""Audit source-only calibration and every selective real prediction."""
import re
import subprocess
import sys
import cv2
import numpy as np
import torch
from . import dino_evidence_gate as E
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

C=E.C;D=E.D;G=E.G;W=E.W;V=E.V;P=E.P;N=E.N


@torch.no_grad()
def main():
    p=E.verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    tests=[C.HERE/n for n in ['test_dino_evidence_gate_model.py','test_dino_wide_visual_model.py','test_dino_wide_model.py','test_dino_localization_model.py','test_heatmap_mode_decoder.py','test_recovery_pseudo_denoise.py']]
    run=subprocess.run([sys.executable,'-m','pytest','-q',*map(str,tests)],capture_output=True,text=True)
    print(run.stdout,flush=True);assert run.returncode==0,run.stderr
    ntests=int(re.search(r'(\d+) passed',run.stdout).group(1));assert ntests==27
    groups=[set(p[k]) for k in ['source_train_rows','calibration_rows','test_rows']]
    assert list(map(len,groups))==[1412,32,32] and not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    source=V.verify()['source_records'];scenario={r['row']:r['scenario'] for r in source}
    assert not {scenario[i] for i in groups[1]}&{scenario[i] for i in groups[2]}
    gates=E.load_gates();cache=C.read(E.DOC/'SOURCE_CACHE.json');calibration_checks={}
    for arm in E.ARMS:
        C.verify(cache['artifacts'][arm])
        with np.load(C.ROOT/cache['artifacts'][arm]['path']) as z:data={k:np.array(z[k]) for k in z.files}
        assert set(data['row'])==set.union(*groups) and set(data['view'])==set(range(5)) and (data['corner']<8).all()
        fit=C.read(E.DOC/f'FIT_{arm}.json');assert fit['steps']==500 and len(fit['history'])==500 and np.isfinite(fit['history']).all()
        train=np.isin(data['row'],p['source_train_rows']);cal=np.isin(data['row'],p['calibration_rows']);test=np.isin(data['row'],p['test_rows'])
        x=torch.as_tensor(data['features'],device='cuda');gate,mean,std,t=gates[arm]
        torch.testing.assert_close(mean,x[train].mean(0),atol=0,rtol=0)
        torch.testing.assert_close(std,x[train].std(0,unbiased=False).clamp_min(1e-4),atol=0,rtol=0)
        scores=gate(E.normalized(x,mean,std)).sigmoid().cpu().numpy()
        best,grid=G.calibrate(scores[cal],data['before'][cal],data['after'][cal],data['view'][cal]==0)
        assert best==fit['calibration'] and grid==fit['calibration_grid'] and t==best['threshold']
        raw=test&(data['view']==0);out=np.where(scores[raw]>=t,data['after'][raw],data['before'][raw])
        assert G.stats(data['before'][raw],out)==fit['source_checks']['test32']
        calibration_checks[arm]=dict(samples=len(data['row']),train=int(train.sum()),calibration=int(cal.sum()),test=int(test.sum()),threshold=t)
    for b in C.read(E.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    for b in C.read(E.DOC/'DECISION_LOCK.json')['fits'].values():C.verify(b)
    original={r['id']:r for r in C.read(E.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    baseline={r['id']:r for r in C.read(E.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    outputs={a:{r['id']:r for r in C.read(E.RAW/f'EVAL_PREDICTIONS_{a}.json')['records']} for a in E.ARMS}
    visual={a:{r['id']:r for r in C.read(V.RAW/f'EVAL_PREDICTIONS_{a}.json')['records']} for a in E.ARMS}
    scores={a:{r['id']:r for r in C.read(E.RAW/f'SCREEN_{a}.json')['metrics']} for a in E.ARMS}
    receipts={r['id']:r for r in C.read(E.RAW/'INFERENCE_RECEIPTS.json')}
    oldreceipts={r['id']:r for r in C.read(V.RAW/'INFERENCE_RECEIPTS.json')}
    for a in E.ARMS:assert set(outputs[a])==set(original)==set(scores[a]) and len(outputs[a])==194
    models=E.heads();backbone,_=D.A.load();accepted={a:0 for a in E.ARMS};changedframes={a:0 for a in E.ARMS};max_delta={a:0. for a in E.ARMS}
    rejected_checks={a:0 for a in E.ARMS}
    for i,r in enumerate(C.read(E.DOC/'EVAL_PROTOCOL.json')['records']):
        C.verify(r['image']);key=r['id'];old=original[key];im=cv2.imread(str(C.ROOT/r['image']['path']))
        x=W.prepare_input(im,old['prediction']);feature=W.extract(backbone,[x])[0]
        assert N.array_sha(feature)==receipts[key]['feature_sha']==oldreceipts[key]['feature_sha']
        f=torch.as_tensor(feature,device='cuda').float()[None];q=torch.as_tensor(x['points'],device='cuda')[None];valid=torch.as_tensor(x['valid'],device='cuda')[None]
        diag=torch.tensor([x['bbox_diagonal']*x['matrix'][0,0]],device='cuda',dtype=torch.float32)
        logits={a:m(f,q,valid) for a,m in models.items()}
        for a in E.ARMS:
            other=next(k for k in E.ARMS if k!=a);feat,new=G.features(logits[a],logits[other],q,valid,diag)
            gate,mean,std,t=gates[a];s=gate(E.normalized(feat,mean,std)).sigmoid()[0].cpu().numpy()
            candidate=N.C.transform_points(new[0].cpu().numpy(),np.linalg.inv(x['matrix']))
            restored,mask=G.choose(x['original_points'],candidate,s,t,x['valid']);d=receipts[key]['decisions'][a]
            np.testing.assert_array_equal(mask,d['accepted']);np.testing.assert_array_equal(s,d['scores'])
            np.testing.assert_array_equal(feat[0].cpu().numpy(),d['features']);np.testing.assert_array_equal(candidate,d['candidate'])
            pred=outputs[a][key]['prediction'];observed=np.asarray(P.top(pred)['keypoints_xy'])
            np.testing.assert_array_equal(restored,observed);P.assert_preserved(old['prediction'],pred)
            np.testing.assert_array_equal(observed[~mask],x['original_points'][~mask])
            np.testing.assert_array_equal(observed[mask],np.asarray(P.top(visual[a][key]['prediction'])['keypoints_xy'])[mask])
            accepted[a]+=int(mask.sum());changedframes[a]+=int(mask.any());rejected_checks[a]+=int((~mask[:8]).sum())
            max_delta[a]=max(max_delta[a],float(np.abs(restored-observed).max()))
        if (i+1)%50==0:print('EVIDENCE_REPRODUCE',i+1,N.E.gpu(),flush=True)
    pe,pop=E.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in original}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    crop={(r['id'],r['GT_corner']):r for r in C.read(W.DOC/'CROP_SUPPORT_DIAGNOSTIC.json')['rows']}
    rows=[];spatial={};cases=[]
    for a in E.ARMS:
        ar=[]
        for key,r in outputs[a].items():
            b=baseline[key];n=scores[a][key];t=targets[key];gt=np.asarray(t.keypoints_xy);q=np.asarray(P.top(r['prediction'])['keypoints_xy'])
            actual=EM.measure(q,gt,t.keypoint_supervision_mask,perms,r['raw_hw'],b['matched'],b['detected'])
            np.testing.assert_allclose(np.asarray(actual['canonical_errors'],float),np.asarray(n['canonical_errors'],float),atol=1e-9,rtol=0,equal_nan=True)
            assert b['matched']==n['matched'] and b['detected']==n['detected']
            if not b['matched']:continue
            oldq=np.asarray(P.top(original[key]['prediction'])['keypoints_xy'])[:8];valid=np.isfinite(oldq).all(-1)&~(oldq==-1).all(-1)
            for j,(before,after) in enumerate(zip(b['canonical_errors'],n['canonical_errors'])):
                if before is None:continue
                distance=float(np.linalg.norm(oldq[valid]-gt[j],axis=-1).min())
                row=dict(id=key,arm=a,GT_corner=j,before=before,after=after,nearest_R0_point_distance=distance,
                    parent_unreachable_within10=crop[(key,j)]['minimum_possible_error_px']['parent']>10,branches=[b['branch'],n['branch']])
                ar.append(row);rows.append(row)
                if before>40 and after<=10:cases.append(row)
        spatial[a]={}
        for d in [10,20,40]:
            ss=[r for r in ar if r['nearest_R0_point_distance']>d and r['before']>20]
            spatial[a][f'no_old_point_within{d}']=dict(hard=len(ss),recovered=sum(r['after']<=10 for r in ss))
    C.freeze(E.RAW/'CORNER_DIAGNOSTICS.json',dict(status='POSTHOC_GT_ONLY_NOT_SELECTION',rows=rows))
    result=C.read(E.DOC/'RESULTS.json')
    C.freeze(E.DOC/'COMPLETION_AUDIT.json',dict(experiment_complete=True,goal_complete=False,tests_passed=ntests,
        source_calibration_reproduced=calibration_checks,source_split_disjoint=True,real_eval_used_for_gate_fit_or_threshold=False,
        predictions_reproduced=388,prediction_reproduction_max_px=max_delta,rejected_corner_bit_exact_checks=rejected_checks,
        accepted_native_corners=accepted,changed_frames=changedframes,spatial_decomposition=spatial,recovered_over40_cases=cases,
        bounded_screen=result['passed'],new_annotations=0,new_tags=0,auto_promoted=False,
        warning='Bounded screen pass is NOT achievement of the large-error objective;80/100px tails and actual-spatial recovery must be reported.',
        evidence=[C.bound(__file__),C.bound(E.RAW/'CORNER_DIAGNOSTICS.json')]+[C.bound(E.DOC/n) for n in
            ['PROTOCOL.json','FIT_SYN.json','FIT_MIX.json','SOURCE_CACHE.json','DECISION_LOCK.json','OUTPUTS_LOCK.json','RESULTS.json','SCORER_ADAPTER.json']]+[C.bound(t) for t in tests]))
    print('EVIDENCE_AUDIT_COMPLETE',dict(spatial=spatial,accepted=accepted,frames=changedframes,cases=cases),flush=True)


if __name__=='__main__':main()
