"""Audit actual fit traces and frozen references; do not retune anything."""
import json
from pathlib import Path
import re
import numpy as np
import yaml
from . import common as C
from scripts.research.pallet_clean19_structured_easyhard_v1 import common as S


def main():
    fits={a:C.read(C.DOC/f'FIT_{a}.json') for a in ('H_PSEUDO','H_MANUAL')}
    for fit in fits.values():
        C.verify(fit['checkpoint']);C.verify(fit['trace']);C.verify(fit['protocol'])
        assert fit['complete'] and fit['steps']==320 and len(fit['epochs'])==5 and fit['initial_state_verified']
    assert fits['H_PSEUDO']['trainable_inventory']==fits['H_MANUAL']['trainable_inventory']
    args={a:yaml.safe_load((C.RAW/'runs'/a/'args.yaml').read_text()) for a in fits}
    for key in args['H_PSEUDO']:
        if key not in ('name','save_dir'):assert args['H_PSEUDO'][key]==args['H_MANUAL'][key],key
    oldargs=C.read(S.DOC/'PREFLIGHT.json')['args']
    for actual in args.values():
        for key,value in oldargs.items():
            if key not in ('name','save_dir','project'):assert actual[key]==value,(key,actual[key],value)
    traces={a:C.read(C.ROOT/f['trace']['path']) for a,f in fits.items()}
    original=C.read(C.ROOT/C.read(S.DOC/'FIT_PLASTIC_S1.json')['trace']['path'])
    hard=0;unchanged=0
    for i,(a,b) in enumerate(zip(traces['H_PSEUDO'],traces['H_MANUAL'])):
        assert a['occ']==b['occ']==i and a['hard']==b['hard']
        for k in ('input_sha256','box_sha256','kind'):assert a[k]==b[k]
        if a['hard']:hard+=1
        else:
            for k in ('input_sha256','target_sha256','box_sha256'):assert a[k]==b[k]==original[i][k]
            unchanged+=1
    assert hard==320 and unchanged==4800
    for a,t in traces.items():
        for e in range(5):
            rr=t[e*1024:(e+1)*1024]
            assert sum(r['kind']=='HARD' for r in rr)==64
            assert sum(r['kind']=='CLEAN' for r in rr)==448
            assert sum(r['kind']=='SYNTH' for r in rr)==512
    C.save(C.DOC/'PAIR_INTEGRITY_MANUAL_VS_PSEUDO.json',dict(passed=True,actual_occurrences=5120,
           hard=hard,unchanged_original_S1=unchanged,same_RGB_box=True,same_inventory=True,
           same_optimizer_args=True,same_init=True,steps={a:f['steps'] for a,f in fits.items()},
           support_gradient_test=C.bind(C.DOC/'PRETRAIN_TESTS.json')))
    ll=C.read(C.DOC/'HARD_LABEL_LOCK.json');tl=C.read(C.DOC/'TEACHER_HARD_PREDICTION_LOCK.json');rl=C.read(C.DOC/'RAW_PREDICTIONS_LOCK.json');pl=C.read(C.DOC/'POSE_DECISIONS_LOCK.json');sc=C.read(C.DOC/'SCORING_START.json')
    assert ll['created_at']<tl['created_at']<rl['created_at']<=pl['created_at']<sc['created_at']
    for lock,key in [(ll,'labels'),(tl,'predictions'),(rl,'predictions'),(pl,'poses')]:C.verify(lock[key])
    res=C.read(C.DOC/'RESULTS.json');assert res['base_reproduced'] and not res['selected_train_recording_intersection']
    for g,n in [('CLEAN',29),('MODERATE',21),('SEVERE',78),('ALL',128)]:
        for arm in res['groups'][g]:assert res['groups'][g][arm]['twoD']['total_frames']==n
    assert C.read(C.DOC/'SOURCE_PRESERVATION.json')['frames']==256
    assert C.read(C.DOC/'VERIFIED_VISIBLE.json')['groups']['ALL']['BASE']['n']==66
    # Public annotations contain provenance/coverage only, never exact manual coordinates.
    for name in ('HARD_LABEL_LOCK.json','HARD_PROVENANCE_PUBLIC.json','HARD_COVERAGE_PUBLIC.json'):
        s=(C.DOC/name).read_text();assert '"xy"' not in s and '"corners"' not in s
    md=(C.DOC/'REPORT_KO.md').read_text()
    for path in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',md):assert (C.DOC/path).is_file(),path
    C.save(C.DOC/'COMPLETION_AUDIT.json',dict(complete=True,created_at=C.now(),actual_fit_pair_pass=True,
           frozen_before_scoring=True,label_lock_before_teacher=True,baseline_reproduced=True,
           original_clean_source_exact=True,manual_coordinate_privacy=True,report_images_present=True,
           training_not_generalization=True,automatic_promotion=False,
           annotation_protocol='USER_APPROVED_PNP_ASSISTED_WITH_SHARED_PNP_BOX',
           code=[C.bind(p) for p in sorted(Path(__file__).parent.glob('*.py'))]))
    C.set_state('COMPLETE',training='COMPLETE',decision=C.read(C.DOC/'DECISION.json')['primary'],
                report='_docs/experiments/pallet_min_hard_ab_v1/REPORT_KO.md',user_action_required=False)
    print('COMPLETION_AUDIT_PASS')


if __name__=='__main__':main()
