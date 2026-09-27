"""Reuse the exact Plastic pseudo pipeline in an isolated Wood namespace."""
from collections import Counter
from unittest.mock import patch
import numpy as np
from . import common as C
from scripts.research.pallet_type_selftrain_v1 import pseudo as P
from scripts.research.pallet_type_selftrain_v1.recovery_pose import paired_labels

def main():
    if not (C.DOC/'PSEUDO_COMPLETE.json').exists():
        def adapter(path):
            data=C.read(path)
            if str(path)==str(C.DOC/'POOL.json'):
                data=dict(data,records=[{k:v for k,v in r.items() if k!='raw_hw'} for r in data['records']])
            return data
        with patch.object(P.C,'DOC',C.DOC),patch.object(P.C,'RAW',C.RAW),patch.object(P.C,'read',adapter):P.run()
    completed=C.read(C.DOC/'PSEUDO_COMPLETE.json')
    for b in completed['artifacts']+[completed['protocol']]:C.verify(b)
    accepted=C.read(C.RAW/'PSEUDO_ACCEPTED.json')
    pool=C.read(C.DOC/'POOL.json')['records']
    frames=[C.read(C.RAW/'pseudo_frames'/f'{r["id"]}.json') for r in pool]
    C.save(C.RAW/'WOOD_RAW_PSEUDO.json',[dict(id=r['id'],raw=r['raw']) for r in frames],True)
    C.save(C.RAW/'WOOD_CORRECTED_PSEUDO.json',[dict(id=r['id'],refined=r['refined']) for r in frames if r['refined'] is not None],True)
    C.save(C.RAW/'WOOD_ACCEPTED_SHARED.json',accepted,True)
    C.save(C.DOC/'WOOD_PSEUDO_PROTOCOL.json',dict(protocol=C.bind(C.DOC/'PSEUDO_PROTOCOL.json'),
        original_code=C.bind(P.__file__),adapter_code=C.bind(__file__),complete=completed,
        raw=C.bind(C.RAW/'WOOD_RAW_PSEUDO.json'),corrected=C.bind(C.RAW/'WOOD_CORRECTED_PSEUDO.json'),shared=C.bind(C.RAW/'WOOD_ACCEPTED_SHARED.json'),
        raw_control_shares_teacher_based_selection=True),True)
    support=[paired_labels(r)[1] for r in accepted]
    sampled=np.random.default_rng(9021).choice(sorted(r['id'] for r in accepted),512,replace=True).tolist() if accepted else []
    # Operational diversity floor, not a power or outcome criterion. Locked
    # before either fit; retains the old512 replacement exposure and both source recordings.
    recordings=Counter(r['recording'] for r in accepted)
    allowed=len(set(sampled))>=64 and len(recordings)==2 and bool(support) and min(support)>=6
    C.save(C.DOC/'POOL_DECISION.json',dict(allowed=allowed,status='FIT_ALLOWED' if allowed else 'WOOD_PSEUDO_POOL_INSUFFICIENT',
        accepted=len(accepted),sampled_unique=len(set(sampled)),recordings=dict(recordings),support_histogram=dict(Counter(support)),
        minimum_rule='At least64 actually sampled unique images (one per optimizer batch within one epoch), both source recordings represented, and>=6 shared supervised points per accepted image; operational feasibility only, not power.',
        comparison=dict(plastic_accepted=249,plastic_sampled_unique=217,real_slots_per_epoch=512,wood_expected_occurrences_per_unique=512/len(set(sampled)) if sampled else None),
        no_eval_scores_used=True,no_threshold_changes=True,artifacts=[C.bind(C.RAW/'WOOD_ACCEPTED_SHARED.json')]),True)
    print('WOOD_POOL_DECISION',len(accepted),len(set(sampled)),dict(recordings),allowed,flush=True)

if __name__=='__main__':main()
