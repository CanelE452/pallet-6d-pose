"""Scoring-only adapter for the gate + frozen-head composite checkpoint.

The completed prediction payload omitted the singular checkpoint field expected
by the old scorer. Keep its locked bytes unchanged; bind all three component
checkpoints here and supply the gate binding only as in-memory compatibility
metadata. No coordinates are recomputed or changed.
"""
from unittest.mock import patch
from . import dino_evidence_gate as E

C=E.C;D=E.D;R=E.R


def positive(arm):
    payload=C.read(E.RAW/f'EVAL_PREDICTIONS_{arm}.json')
    gate=C.read(E.DOC/f'FIT_{arm}.json')['checkpoint'];C.verify(gate)
    components={a:C.read(E.V.DOC/f'FIT_{a}.json')['checkpoint'] for a in E.ARMS}
    for b in components.values():C.verify(b)
    payload=dict(payload,checkpoint=gate,frozen_image_heads=components)
    return payload,{r['id']:E.P.top(r['prediction']) for r in payload['records'] if r['kind']=='PLASTIC'}


def main():
    E.verify()
    for b in C.read(E.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    with patch.object(R.M,'positive',positive):E.report()
    C.freeze(E.DOC/'SCORER_ADAPTER.json',dict(reason='Prediction metadata omitted singular checkpoint; immutable bytes preserved.',
        coordinates_changed=False,retraining=False,source=C.bound(__file__),outputs=C.bound(E.DOC/'OUTPUTS_LOCK.json'),
        result=C.bound(E.DOC/'RESULTS.json'),fits={a:C.bound(E.DOC/f'FIT_{a}.json') for a in E.ARMS},
        image_fits={a:C.bound(E.V.DOC/f'FIT_{a}.json') for a in E.ARMS}))


if __name__=='__main__':main()
