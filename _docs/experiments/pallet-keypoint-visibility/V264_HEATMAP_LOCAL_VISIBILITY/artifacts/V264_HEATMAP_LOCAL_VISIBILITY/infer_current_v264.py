"""Current frozen-token inference only after both SYN finals/config are sealed."""
from pathlib import Path
import io
import json
import sys
import types
import numpy as np
import torch

HERE = Path(__file__).absolute().parent


def main():
    path = HERE / 'v264_model.py'
    for a in reversed((path,*path.parents)):
        assert not a.is_symlink()
    m = types.ModuleType('v264_current_model'); m.__file__ = str(path);sys.modules[m.__name__]=m
    exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__)
    freeze = json.loads(m.safe(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json').read_text())
    assert freeze['completed'] and freeze['PASS'] and freeze['threshold_E_logit'] == 0
    assert m.binding(path) == freeze['model_code']
    protocol=json.loads(m.read_bound(freeze['protocol']))
    assert m.binding(Path(__file__).absolute()) == protocol['inference_code']
    out=m.safe(HERE/'current_inference_v1'); assert not out.exists();out.mkdir()
    cp=m.checked(m.ROOT/'V263_SPATIAL_HEAD_C/current223_frozen_inference_v1/COMPLETE.json',
                  '8fcfa2761b26e3f2fa236f525bf98267d4964d7a650fef5bceb99b7d2ac3e881')
    packet=json.loads(cp.read_text())
    for name in ('SPATIAL_TOKENS_FP16.npy','TOKEN_SUPPORT.npy','INPUT_ROWS1784.jsonl'):
        b=packet['outputs'][name];m.checked(b['path'],b['sha256'],b['size_bytes'])
    inp=[json.loads(line) for line in m.read_bound(packet['outputs']['INPUT_ROWS1784.jsonl']).splitlines()]
    assert len(inp)==1784 and all(r['global_index']==i for i,r in enumerate(inp))
    device=m.runtime(); frozen=m.frozen_heatmap_model(device)
    tokens=np.load(m.safe(packet['outputs']['SPATIAL_TOKENS_FP16.npy']['path']),mmap_mode='r',allow_pickle=False)
    support=np.load(m.safe(packet['outputs']['TOKEN_SUPPORT.npy']['path']),allow_pickle=False)
    native=np.array([r['native_slot'] for r in inp],np.int64)
    features=m.pool_features(frozen,tokens,support,native,device)
    available=features['available']&np.array([r['available'] for r in inp],bool)
    assert available.sum()==1739
    np.savez_compressed(out/'CURRENT_POOLED_FEATURES.npz',**features)
    predictions={}
    for arm in m.ARMS:
        classifier=m.VisibilityReadout().to(device)
        state=torch.load(io.BytesIO(m.read_bound(freeze['checkpoints'][arm])),map_location='cpu')
        classifier.load_state_dict(state['state_dict'],strict=True)
        logit,pred=m.infer(classifier,m.design(features,arm),available,device)
        predictions[arm]=(logit,pred)
    records=[]
    for i,row in enumerate(inp):
        r={'global_index':i,'row_id':row['frame_id']+':native'+str(row['native_slot']),
           'frame_id':row['frame_id'],'native_slot':row['native_slot'],'available':bool(available[i])}
        for arm in m.ARMS:
            logit=float(predictions[arm][0][i]); pred=int(predictions[arm][1][i])
            r[arm]={'E_logit':logit,'p_E':float(1/(1+np.exp(-np.clip(logit,-700,700)))),
                    'pred':{0:'D',1:'E',-1:'U'}[pred]}
        records.append(r)
    p=HERE/'CURRENT_ALL1784_PREDICTIONS.jsonl';assert not p.exists()
    p.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
    m.write_json(HERE/'CURRENT_PREDICTIONS_FROZEN.json',{
        'completed':True,'PASS':True,'configuration':m.binding(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json'),
        'predictions':m.binding(p),'feature_packet':m.binding(out/'CURRENT_POOLED_FEATURES.npz'),
        'code':m.binding(Path(__file__).absolute()),'current_source_complete':m.binding(cp),
        'rows':1784,'supported':int(available.sum()),'current_GT_or_annotation_reads':0,
        'new_optimizer_updates':0,'new_backbone_forwards':0,'new_frozen_spatial_forwards':1784,
        'new_PnP':0,'location_coordinates_unchanged':True})
    print(json.dumps({'completed':True,'freeze':m.binding(HERE/'CURRENT_PREDICTIONS_FROZEN.json')}),flush=True)


if __name__=='__main__':
    main()
