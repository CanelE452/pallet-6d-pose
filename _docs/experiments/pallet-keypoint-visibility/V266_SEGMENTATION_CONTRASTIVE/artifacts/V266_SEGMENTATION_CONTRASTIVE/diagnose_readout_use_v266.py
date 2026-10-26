"""Post-freeze descriptor interventions; diagnostics only, no adopted decision rule."""
from pathlib import Path
import hashlib
import io
import json
import sys
import types
import numpy as np
import torch

HERE=Path(__file__).absolute().parent


def safe(value):
    p=Path(value)
    assert p.is_absolute() and not any(x in str(p).lower() for x in ('v112','cache','.pyc','..')),str(p)
    for a in reversed((p,*p.parents)):assert not a.is_symlink(),str(a)
    return p


def read(binding):
    p=safe(binding['path']);raw=p.read_bytes()
    assert len(raw)==binding['size_bytes'] and hashlib.sha256(raw).hexdigest()==binding['sha256']
    return raw


def main():
    torch.set_num_threads(1)
    freeze=json.loads(safe(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json').read_bytes())
    receipt=json.loads(safe(HERE/'CURRENT_PREDICTIONS_FROZEN.json').read_bytes())
    code=read(freeze['model_code'])
    m=types.ModuleType('v266_postfreeze_usage');m.__file__=freeze['model_code']['path'];sys.modules[m.__name__]=m
    exec(compile(code,m.__file__,'exec'),m.__dict__)
    z=np.load(io.BytesIO(read(receipt['descriptor_predictions'])),allow_pickle=False)
    available=z['available'];reports={}
    for arm in m.ARMS:
        model=m.OwnerVisibilityResidual().eval()
        cp=torch.load(io.BytesIO(read(freeze['checkpoints'][arm])),map_location='cpu')
        model.load_state_dict(cp['state_dict'],strict=True)
        original=np.array(z[arm],copy=True)
        variants={'actual':original}
        uniform=original.copy();uniform[:,1:10]=1/3;variants['uniform_owner']=uniform
        for cls,name in enumerate(('all_target','all_foreign','all_background')):
            v=original.copy();v[:,1:10]=np.tile(np.eye(3)[cls],3);variants[name]=v
        scores={}
        with torch.inference_mode():
            for name,array in variants.items():
                x=torch.as_tensor(array).float();scores[name]=(x[:,0]+model.residual(x)[:,0]).numpy()
        use=available
        result={}
        for name,score in scores.items():
            delta=score-scores['actual']
            result[name]=dict(mean_abs_logit_change=float(np.abs(delta[use]).mean()),
                max_abs_logit_change=float(np.abs(delta[use]).max()),
                supported_label_changes=int(((score>0)!=(scores['actual']>0))[use].sum()))
        effect=scores['all_foreign']-scores['all_target']
        result['all_foreign_minus_all_target']=dict(mean=float(effect[use].mean()),
            median=float(np.median(effect[use])),min=float(effect[use].min()),max=float(effect[use].max()),
            positive_fraction=float((effect[use]>0).mean()),
            interpretation='Positive moves E score upward when owner descriptors alone change target->foreign; never an E ground-truth prediction.')
        case={}
        for i in (256,370,908,996):
            case[str(i)]=dict(actual_E_logit=float(scores['actual'][i]),base_E_logit=float(original[i,0]),
                owner_q=original[i,1:4].tolist(),owner_heatmap=original[i,4:7].tolist(),owner_global=original[i,7:10].tolist(),
                all_foreign_E_logit=float(scores['all_foreign'][i]),all_target_E_logit=float(scores['all_target'][i]),
                uniform_owner_E_logit=float(scores['uniform_owner'][i]))
        reports[arm]=dict(interventions=result,predefined_cases=case)
    out=safe(HERE/'readout_diagnostic_v1');assert not out.exists();out.mkdir()
    result=dict(completed=True,PASS=True,descriptor_only_interventions=True,training_or_threshold_changes=0,
        current_annotation_or_RGB_reads=0,supported_queries=int(available.sum()),owner_class_order=list(m.OWNER_CLASSES),
        caveat='Synthetic descriptor interventions are out-of-distribution counterfactuals. They test this frozen residual readout response, not physical visibility or a deployable replacement. No variant is chosen by real results.',arms=reports)
    with safe(out/'RESULTS.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
