"""Count visual support in existing prepared-image padding without changing it."""
from pathlib import Path
import json
import numpy as np
from . import verify_train_inputs as A


def main():
    p=A.read(A.DOC/'INPUT_PROTOCOL.json')
    receipt=A.read(A.RAW/'TRAIN_APPEARANCE_INPUTS.json')
    for binding in [receipt['descriptors'],p['inputs']['metadata'],p['inputs']['poses']]:
        A.verify(binding)
    rows_all=A.read(A.ROOT/p['inputs']['metadata']['path'])
    poses=A.read(A.ROOT/p['inputs']['poses']['path'])
    with np.load(A.ROOT/receipt['descriptors']['path'],allow_pickle=False) as z:
        indices=z['source_index'];rows=[rows_all[i] for i in indices]
        assert len(rows)==2598 and all(r['split']=='TRAIN' for r in rows)
        matrices=z['crop_matrices']
        masks={m:z[m+'_valid'] for m in A.MODELS}
        support={m:z[m+'_support8'] for m in A.MODELS}
    counts={m:dict(valid_candidates=0,supported_points=0,padding_supported_points=0,
                   candidates_with_padding_support=0,frames_with_padding_support=0) for m in A.MODELS}
    for j,row in enumerate(rows):
        pad=int(row['pad']);h,w=row['hw']
        assert pad==row['pad'] and 0<=pad<min(h,w)/2
        for m in A.MODELS:
            touched=False
            hypotheses={x['name']:x for x in poses['records'][m][row['id']]['hypotheses']}
            for k,name in enumerate(A.HYP):
                if not masks[m][j,k]:continue
                uv,_,supported=A.project(hypotheses[name]['pose'],row['K'],matrices[j],row['hw'])
                np.testing.assert_array_equal(supported,support[m][j,k])
                native=(uv[:,0]>=pad)&(uv[:,0]<w-pad)&(uv[:,1]>=pad)&(uv[:,1]<h-pad)
                padding=supported&~native
                counts[m]['valid_candidates']+=1
                counts[m]['supported_points']+=int(supported.sum())
                counts[m]['padding_supported_points']+=int(padding.sum())
                counts[m]['candidates_with_padding_support']+=int(padding.any())
                touched|=bool(padding.any())
            counts[m]['frames_with_padding_support']+=int(touched)
    result=dict(complete=True,PASS=True,scope='TRAIN input-only prepared-padding diagnostic; unchanged feature contract',
        code=A.bind(Path(__file__)),projection_operator=A.bind(Path(A.__file__)),
        protocol=A.bind(A.DOC/'INPUT_PROTOCOL.json'),descriptors=receipt['descriptors'],
        metadata=p['inputs']['metadata'],poses=p['inputs']['poses'],frames=2598,
        original_valid_candidates=20776,padding_values=sorted(set(r['pad'] for r in rows)),models=counts,
        padding_definition='Supported projected corner outside [pad,width-pad) x [pad,height-pad) of the prepared canvas.',
        interpretation='Prepared source RGB includes reflected padding. A supported token location can lie in that padding and is not necessarily a location in the original camera image. Even tokens sampled inside the original region have contextual receptive fields.',
        original_support_mask_changed=False,descriptor_values_changed=False,normalization_changed=False,
        new_fits=0,new_forwards=0,new_pose_solves=0,target_values_read=0,performance_improvement_measured=False)
    out=A.DOC/'PADDING_SUPPORT_AUDIT.json'
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print('PADDING_SUPPORT_AUDIT',json.dumps(counts),flush=True)


if __name__=='__main__':main()
