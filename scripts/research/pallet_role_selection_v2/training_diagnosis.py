from common import *
from model import SelectorResidual,PERMS
import torch,numpy as np,collections

def main():
    torch.set_num_threads(2);z=torch.load(PREVIOUS/'train.pt',weights_only=True)
    gt=np.array([r['gt'] for r in read(PREVIOUS/'train_records.json')]);p=z['stock'].numpy()[:,np.array(PERMS),:][:,:,:8]
    e=np.linalg.norm(p-gt[:,None],axis=-1).mean(-1);fix=e[:,0]-e.min(-1)>20;results={}
    for kind in ['robust','mean']:
        for seed in [1,2,3]:
            path=OUT/'models'/f'C_select_residual_s{seed}'/'final.pt' if kind=='robust' else OUT/'mean_risk_followup'/'models'/f'seed{seed}.pt'
            m=SelectorResidual('C_select_residual',seed);m.load_state_dict(torch.load(path,weights_only=True));m.eval()
            with torch.no_grad():o=m(z)
            choice=o['choice'].numpy();chosen=e[np.arange(len(e)),choice];risk=(o['risk']*o['scale'][:,None]).numpy()
            results[f'{kind}_s{seed}']=dict(choice_counts={str(k):v for k,v in collections.Counter(choice).items()},
                big_fixable_repaired=int(((e[:,0]-chosen>20)&fix).sum()),mean_selection_regret_px=float((chosen-e.min(-1)).mean()),
                mean_predicted_risks_px=risk.mean(0).tolist(),true_mean_risks_px=e.mean(0).tolist())
    write(OUT/'TRAINING_DIAGNOSIS.json',dict(train_frames=len(e),includes_detector_misses=True,
        GT_best_counts={str(k):v for k,v in collections.Counter(e.argmin(-1)).items()},large_fixable=int(fix.sum()),results=results,
        note='Training-only diagnostic; no model selection or further sweep from these values.'))
    print(json.dumps(read(OUT/'TRAINING_DIAGNOSIS.json'),indent=2))
if __name__=='__main__':main()
