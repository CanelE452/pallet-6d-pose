"""CPU source replay, exhaustive source frontier and exact null-output audit."""
import re
import subprocess
import sys
import numpy as np
import torch
from . import dino_joint_translate as T

C=T.C


def feasible(raw,combined):
    return raw['PCK10']>=raw['input_PCK10']-.01 and raw['PCK20']>=raw['input_PCK20'] and raw['P90']<=1.1*raw['input_P90'] and raw['damaged']<=.01*raw['good'] and combined['beneficial_precision']>=.95


@torch.no_grad()
def main():
    p=T.verify();torch.set_num_threads(2)
    tests=[C.HERE/n for n in ['test_dino_joint_translate_model.py','test_dino_multipeak_model.py','test_dino_source_diversity.py','test_dino_visual_long.py','test_dino_wide_visual_model.py','test_dino_wide_model.py','test_dino_localization_model.py','test_heatmap_mode_decoder.py','test_recovery_pseudo_denoise.py']]
    run=subprocess.run([sys.executable,'-m','pytest','-q',*map(str,tests)],capture_output=True,text=True);print(run.stdout,flush=True)
    assert run.returncode==0,run.stderr;ntests=int(re.search(r'(\d+) passed',run.stdout).group(1));assert ntests==39
    fit=C.read(T.DOC/'FIT_REGISTER.json');C.verify(fit['source_proposals']);rows=C.read(T.RAW/'SOURCE_PROPOSALS.json')['records'];assert len(rows)==320
    adapter=C.read(T.DOC/'CPU_SOURCE_ADAPTER.json');C.verify(adapter['code']);C.verify(adapter['source_proposals'])
    models={}
    for a in T.D.ARMS:
        f=C.read(T.X.DOC/f'FIT_{a}.json');C.verify(f['checkpoint']);m=T.V.V.Head()
        m.load_state_dict(torch.load(C.ROOT/f['checkpoint']['path'],map_location='cpu',weights_only=False)['model']);models[a]=m.eval()
    side=np.load(T.SIDECAR);src=T.P.SourceData();np.testing.assert_array_equal(side['record_index'],src.data.indices)
    cache={r.get('row'):r for r in C.read(T.X.DOC/'CACHE_COMPLETE.json')['records'] if r['domain']=='source'}
    lookup={(r['row'],r['view']):r for r in rows};assert len(lookup)==320
    for i,row in enumerate(p['calibration_rows']+p['test_rows']):
        meta=cache[row];C.verify(meta['cache'])
        with np.load(C.ROOT/meta['cache']['path']) as z:r=dict(meta,**{k:np.array(z[k]) for k in z.files if k!='protocol_sha256'})
        t=torch.as_tensor(r['feature']).float()[None];q0=torch.as_tensor(r['points'])[None];v0=torch.as_tensor(r['valid'])[None]
        zz={a:m(t,q0,v0) for a,m in models.items()};lp=T.M.logmass(zz['SYN'],zz['MIX'])
        assert torch.isfinite(lp).all() and lp.min()>=np.log(1e-12)-1e-5 and lp.max()<=1e-5
        perms=side['permutations'][row,:int(side['order'][row])]
        for j in range(5):
            q,v=T.view(r,j);proposal=T.M.propose(lp,torch.as_tensor(q),torch.as_tensor(v));stored=lookup[(row,j)]
            assert proposal==stored['proposal'];assert proposal['gain']<64
            out=T.M.restore(q,proposal,np.eye(3));error=T.J.canonical_errors(np.stack([q,out]),r['target'],r['target_valid'],perms,r['matrix'][0,0])
            np.testing.assert_allclose(error,np.asarray(stored['errors'],float),atol=0,rtol=0,equal_nan=True)
            np.testing.assert_array_equal(out[8],q[8])
            if proposal['available']:
                perm=np.asarray(T.M.PERMS[proposal['permutation']])
                np.testing.assert_allclose(out[:8]-q[perm[:8]],np.tile(proposal['delta_crop'],(8,1)),atol=5e-5,rtol=0)
        if (i+1)%16==0:print('REGISTER_SOURCE_REPRODUCED',i+1,flush=True)
    cal=[r for r in rows if r['row'] in p['calibration_rows']];raw=[r for r in cal if r['view']==0]
    for entry in fit['grid']:
        threshold=entry['threshold'];s=T.stats(raw,threshold);combined=T.stats(cal,threshold)
        assert s==entry['raw'] and combined==entry['combined'] and feasible(s,combined)==entry['feasible']
    # Exhaustive source-only threshold frontier checks whether the finite grid
    # caused the null outcome. Diagnostic only: DO NOT change the locked policy.
    thresholds=sorted({0.,1e9,*[r['proposal']['gain'] for r in cal if T.M.changed(r['proposal'])]})
    frontier=[]
    for threshold in thresholds:
        s=T.stats(raw,threshold);combined=T.stats(cal,threshold)
        frontier.append(dict(threshold=threshold,raw=s,combined=combined,feasible=feasible(s,combined)))
    nonempty=[r for r in frontier if r['feasible'] and r['combined']['selected']>0]
    best=max(nonempty,key=lambda r:(r['raw']['recovered'],r['combined']['recovered'],r['threshold'])) if nonempty else None
    C.freeze(T.DOC/'EXACT_SOURCE_FRONTIER_DIAGNOSTIC.json',dict(status='SOURCE_ONLY_NOT_APPLIED',thresholds=frontier,
        nonempty_feasible=len(nonempty),best=best,locked_threshold=fit['threshold'],actual_policy_changed=False,
        raw_test_unfiltered=T.stats([r for r in rows if r['row'] in p['test_rows'] and r['view']==0],0.)))
    for b in C.read(T.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    shortcut=C.read(T.DOC/'NULL_POLICY_SHORTCUT.json');C.verify(shortcut['code']);C.verify(shortcut['source_fit'])
    assert fit['threshold']==1e9>shortcut['conservative_float_bound']>shortcut['mathematical_gain_bound']
    original={r['id']:r for r in C.read(T.D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    predictions=C.read(T.RAW/f'EVAL_PREDICTIONS_{T.ARM}.json');assert len(predictions['records'])==194
    assert not predictions['registration_proposals_computed'] and predictions['real_neural_forwards']==0
    for r in predictions['records']:assert r==original[r['id']]
    receipts=C.read(T.RAW/'INFERENCE_RECEIPTS.json');assert len(receipts)==194 and all(not r['applied'] and not r['proposal_computed'] for r in receipts)
    result=C.read(T.DOC/'RESULTS.json');assert result['summary']==result['R0_summary'] and result['applied_frames']==0 and result['recovery']['recovered']==result['recovery']['damaged']==0
    assert not torch.cuda.is_initialized()
    C.freeze(T.DOC/'COMPLETION_AUDIT.json',dict(bounded_protocol_complete=True,goal_complete=False,tests_passed=ntests,
        source_cases_reproduced_exactly=320,source_backend='CPU_FLOAT32_THREADS2',source_calibration_grid_reproduced=True,
        real_R0_predictions_preserved_exactly=194,real_registration_search_executed=False,new_real_GPU_forwards=0,
        null_shortcut_verified=True,normal_preservation_is_not_improvement=True,exact_source_nonempty_feasible=len(nonempty),
        evidence=[C.bound(__file__)]+[C.bound(t) for t in tests]+[C.bound(T.DOC/f) for f in
            ['PROTOCOL.json','CPU_SOURCE_ADAPTER.json','FIT_REGISTER.json','DECISION_LOCK.json','NULL_POLICY_SHORTCUT.json','OUTPUTS_LOCK.json','RESULTS.json','EXACT_SOURCE_FRONTIER_DIAGNOSTIC.json']]))
    print('REGISTER_AUDIT_COMPLETE',dict(nonempty_exact_source_thresholds=len(nonempty),best=best,real_changes=0,goal_complete=False),flush=True)


if __name__=='__main__':main()
