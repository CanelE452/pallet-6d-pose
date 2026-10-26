"""Portable CPU replay: all1784 residuals and32 full dense forwards per new arm.

Uses only files in this package, no original corpus, real annotations, RGB,
large frozen encoder, training optimizer or PnP. It preserves sealed logit0.
"""
from pathlib import Path
import argparse
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
    for a in reversed((p,*p.parents)):
        assert not a.is_symlink(),str(a)
    return p


def bound_read(binding):
    # Relocate a source-root binding into an unchanged packaged relative layout.
    original=Path(binding['path'])
    marker='V266_SEGMENTATION_CONTRASTIVE'
    assert marker in original.parts
    relative=Path(*original.parts[original.parts.index(marker)+1:])
    p=safe(HERE/relative);raw=p.read_bytes()
    assert len(raw)==binding['size_bytes'] and hashlib.sha256(raw).hexdigest()==binding['sha256'],str(p)
    return raw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='prediction_replay_v1')
    args=parser.parse_args();assert Path(args.out).name==args.out and '..' not in args.out
    out=safe(HERE/args.out);assert not out.exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    freeze=json.loads(safe(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json').read_bytes())
    receipt=json.loads(safe(HERE/'CURRENT_PREDICTIONS_FROZEN.json').read_bytes())
    assert freeze['PASS'] and freeze['threshold_E_logit']==0
    assert json.loads(bound_read(receipt['configuration']))==freeze
    code=bound_read(freeze['model_code'])
    m=types.ModuleType('v266_portable_model');m.__file__=str(HERE/'v266_model.py');sys.modules[m.__name__]=m
    exec(compile(code,m.__file__,'exec'),m.__dict__)
    preds=[json.loads(x) for x in bound_read(receipt['predictions']).splitlines()]
    descriptor=np.load(io.BytesIO(bound_read(receipt['descriptor_predictions'])),allow_pickle=False)
    fixture=np.load(io.BytesIO(bound_read(receipt['forward_replay_inputs'])),allow_pickle=False)
    maps=np.load(io.BytesIO(bound_read(receipt['dense_predictions'])),allow_pickle=False)
    assert len(preds)==1784 and descriptor['available'].sum()==1739
    results={}
    with torch.inference_mode():
        for arm in m.ARMS:
            model=m.OwnerVisibilityResidual().eval()
            cp=torch.load(io.BytesIO(bound_read(freeze['checkpoints'][arm])),map_location='cpu')
            model.load_state_dict(cp['state_dict'],strict=True)
            x=torch.as_tensor(descriptor[arm]).float()
            logit=(x[:,0]+model.residual(x)[:,0]).numpy()
            reference=np.array([r[arm]['E_logit'] for r in preds])
            delta=float(np.max(np.abs(logit-reference)))
            classification=np.where(descriptor['available'],np.where(logit>0,'E','D'),'U')
            assert np.array_equal(classification,np.array([r[arm]['pred'] for r in preds]))
            assert delta<1e-4
            full=model(torch.as_tensor(fixture['tokens']).float(),torch.as_tensor(fixture['support']).bool(),
                torch.as_tensor(fixture['auxiliary']).float(),torch.as_tensor(fixture['uv']).float(),
                torch.as_tensor(fixture['base_E_logit']).float())
            full_delta=float(np.max(np.abs(full['E_logit'].numpy()-fixture[arm+'_E_logit'])))
            dense_delta=float(np.max(np.abs(full['dense_probabilities'].numpy()-maps[arm][fixture['source_index']])))
            assert full_delta<1e-4 and dense_delta<1e-4
            results[arm]=dict(all1784_labels_identical=True,descriptor_logit_max_abs_diff=delta,
                full_dense_forward_rows=len(fixture['source_index']),full_forward_logit_max_abs_diff=full_delta,
                full_forward_dense_probability_max_abs_diff=dense_delta)
    out.mkdir()
    result=dict(completed=True,PASS=True,arms=results,optimizer_updates=0,source_or_RGB_or_annotation_reads=0,
        real_model_selection=0,new_backbone_or_localization_forward=0,PnP=0,
        runtime=dict(torch=torch.__version__,numpy=np.__version__),
        limitation='Full dense branch forward is checked on32 fixed input-only rows perarm; all1784 classification outputs replay from sealed22D descriptors. All1784 dense maps are hash checked, not re-forwarded without the omitted350MB token file.')
    with safe(out/'COMPLETE.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
