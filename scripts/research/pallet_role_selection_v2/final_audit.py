from common import *
from model import SelectorResidual,PERMS
import torch,numpy as np

def main():
    torch.set_num_threads(2);checks=dict(initial_audit=read(OUT/'AUDIT.json')['PASS']);allhashes={r['sha256'] for r in read(PREVIOUS/'RENDER_COMPLETE.json')['rows']+read(OUT/'RENDER_COMPLETE.json')['rows']}
    replay=[];metric=[];bounds=[]
    for sub in ['mean_risk_followup','foreground_matched']:
        folder=OUT/sub;rows=read(folder/'test_records.json');z=torch.load(folder/'test.pt',weights_only=True)
        hashes={r['sha256'] for r in rows};assert not hashes&allhashes;allhashes|=hashes
        predictions=np.load(folder/'predictions.npz');rr=read(folder/'PER_FRAME.json');choices=np.load(folder/'choices.npz')
        gt=np.array([r['gt'] for r in rows]);matched=np.array([r['matched'] for r in rr['stock']]);diag=np.array([np.linalg.norm(r['raw_hw']) for r in rows])
        for name in predictions.files:
            assert np.isfinite(predictions[name]).all();assert np.array_equal(matched,[r['matched'] for r in rr[name]])
            expected=np.minimum(1,np.where(matched,np.linalg.norm(predictions[name][:,:8]-gt,axis=-1).mean(-1)/diag,1))
            metric.append(float(np.abs(expected-np.array([r['normalized'] for r in rr[name]])).max()))
            if name in ['stock','canonical']:continue
            kind,seed=name.rsplit('_s',1);seed=int(seed)
            m=SelectorResidual('A_direct' if kind=='direct' else 'C_select_residual',seed)
            if sub=='mean_risk_followup':
                path=folder/'models'/f'seed{seed}.pt' if kind=='mean' else OUT/'models'/f'{"A_direct" if kind=="direct" else "C_select_residual"}_s{seed}'/'final.pt'
            else:path=OUT/'mean_risk_followup'/'models'/f'seed{seed}.pt' if kind=='unmatched_training' else folder/'models'/f'{kind}_s{seed}.pt'
            m.load_state_dict(torch.load(path,weights_only=True));m.eval()
            with torch.no_grad():p=m(z)
            replay.append(float(np.abs(p['points'].numpy()-predictions[name]).max()))
            assert np.array_equal(p['choice'].numpy(),choices[name])
            base=z['stock'].numpy()[np.arange(len(rows))[:,None],np.array(PERMS)[choices[name]]]
            bounds.append(float(np.abs(predictions[name]-base).max()))
        assert read(folder/'RENDER_COMPLETE.json')['gt_projection_max_px']<1e-8
    checks.update(all_new_scene_hashes_disjoint=True,all_detector_populations_identical=True,
        prediction_replay=max(replay)==0,independent_metric_replay=max(metric)<1e-8,residual_bounds=max(bounds)<=16.0001,
        geometry_projection=True,parent_unchanged=all(sha(ROOT/p)==h for p,h in read(PREVIOUS/'COMPLETION.json')['files'].items()))
    write(OUT/'FINAL_AUDIT.json',dict(PASS=all(checks.values()),checks=checks,prediction_replay_max_px=max(replay),metric_replay_max=max(metric),residual_abs_max_px=max(bounds),
        new_observations=1728,models_trained=21,optimizer_steps=25200,real_evaluation=False))
    assert all(checks.values()),checks;print(checks)
if __name__=='__main__':main()
