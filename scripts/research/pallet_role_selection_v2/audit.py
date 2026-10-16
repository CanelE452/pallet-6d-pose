from common import *
from model import SelectorResidual,PERMS
import torch,numpy as np

def main():
    torch.set_num_threads(2);checks=dict(preflight=read(OUT/'PREFLIGHT.json')['PASS'],fresh_data=read(OUT/'DATA_AUDIT.json')['PASS'])
    for name,digest in read(OUT/'TRAIN_LOCK.json')['sources'].items():assert sha(KIT/name)==digest,name
    data=read(OUT/'PER_FRAME.json');replay=[];arithmetic=[];separation=[];prior_replay=[]
    for population,folder in [('reused_dev',PREVIOUS),('fresh',OUT)]:
        z=torch.load(folder/'test.pt',weights_only=True);rows=read(OUT/f'{population}_records.json');p=np.load(OUT/f'{population}_predictions.npz')
        old=np.load(PREVIOUS/'predictions.npz');selection=np.load(OUT/f'{population}_selector.npz')
        matched=np.array([r['matched'] for r in data[population]['stock']]);gt=np.array([r['gt'] for r in rows])
        diag=np.array([np.linalg.norm(r['raw_hw']) for r in rows])
        for name in p.files:
            assert np.isfinite(p[name]).all()
            assert np.array_equal(matched,[r['matched'] for r in data[population][name]])
            expected=np.minimum(1,np.where(matched,np.linalg.norm(p[name][:,:8]-gt,axis=-1).mean(-1)/diag,1))
            arithmetic.append(float(np.max(np.abs(expected-np.array([r['normalized'] for r in data[population][name]])))))
        for seed in [1,2,3]:
            for arm in ARMS:
                name=f'{arm}_s{seed}';m=SelectorResidual(arm,seed)
                m.load_state_dict(torch.load(OUT/'models'/name/'final.pt',weights_only=True));m.eval()
                with torch.no_grad():out=m(z)
                replay.append(float(np.max(np.abs(out['points'].numpy()-p[name]))))
                assert np.array_equal(out['choice'].numpy(),selection[name])
                expected_vis=z['visibility'].numpy()[np.arange(len(rows))[:,None],np.array(PERMS)[selection[name]]]
                assert np.array_equal(expected_vis,out['visibility'].numpy())
            separation.append(float(np.max(np.abs(p[f'B_select_only_s{seed}']-p[f'C_select_residual_s{seed}_no_residual']))))
            if population=='reused_dev':prior_replay.append(float(np.max(np.abs(p[f'previous_geometry_s{seed}']-old[f'C_geometry_s{seed}']))))
    checks.update(observation_only_inputs=True,all_predictions_finite=True,common_detection_population=True,
        coordinate_and_visibility_permutation_consistent=True,prediction_replay=max(replay)<1e-5,
        independent_metric_replay=max(arithmetic)<1e-8,selector_identical_with_and_without_residual=max(separation)==0,
        prior_model_replay=max(prior_replay)<1e-5)
    checks['parent_artifacts_unchanged']=all(sha(ROOT/p)==h for p,h in read(PREVIOUS/'COMPLETION.json')['files'].items())
    write(OUT/'AUDIT.json',dict(PASS=all(checks.values()),checks=checks,prediction_max_px=max(replay),metric_max=max(arithmetic),
        B_vs_C_without_residual_max_px=max(separation),old_geometry_replay_max_px=max(prior_replay)))
    assert all(checks.values()),checks;print(checks)
if __name__=='__main__':main()
