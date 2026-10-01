"""Publish lossless small scorer parameters as inspectable JSON, not pickle."""
from . import common as C
import json
import numpy as np
import torch


def main():
    done=C.read(C.DOC/'TRAINING_COMPLETE.json');assert done['complete'] and done['fit_count']==6
    exports=[]
    for binding in done['fits']:
        C.verify(binding);receipt=C.read(C.ROOT/binding['path']);C.verify(receipt['checkpoint'])
        ck=torch.load(C.ROOT/receipt['checkpoint']['path'],map_location='cpu',weights_only=False)
        state={key:value.detach().cpu().numpy() for key,value in ck['state'].items()}
        out=dict(schema='inspectable_linear94_parameters_v1',arm=ck['arm'],seed=ck['seed'],variant=ck['variant'],d=ck['d'],
            candidate_names=ck['candidate_names'],state={k:v.tolist() for k,v in state.items()},
            mean=ck['mean'].tolist(),std=ck['std'].tolist(),dtype='float32',
            source_checkpoint=receipt['checkpoint'],training_protocol=ck['protocol'],
            adoption_status='SOURCE_VAL_ALL_SEED_GATE_FAILED; real performance untested; not adopted')
        # Round-trip the actual JSON serialization used for publication.
        restored=json.loads(json.dumps(out,allow_nan=False))
        for key,value in state.items():np.testing.assert_array_equal(np.asarray(restored['state'][key],dtype=np.float32),value)
        for key in ['mean','std']:np.testing.assert_array_equal(np.asarray(restored[key],dtype=np.float32),ck[key])
        path=C.DOC/'model_parameters'/f"{ck['arm']}_s{ck['seed']}.json"
        C.save(path,out);exports.append(C.bind(path))
    C.save(C.DOC/'MODEL_EXPORTS.json',dict(complete=True,exports=exports,count=6,lossless_float32_roundtrip=True,
        training_complete=C.bind(C.DOC/'TRAINING_COMPLETE.json'),code=C.bind(__file__)))
    print('SIX_SCORERS_EXPORTED_LOSSLESS_JSON',flush=True)


if __name__=='__main__':main()
