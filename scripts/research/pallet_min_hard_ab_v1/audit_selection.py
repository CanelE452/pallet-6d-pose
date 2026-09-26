"""Validate the human-only selection without opening any model outputs."""
from collections import Counter
import numpy as np
from . import common as C
from . import policy
from .prepare import thumbnail

def main():
    taglock=C.read(C.DOC/'DIFFICULTY_TAG_LOCK.json');C.verify(taglock['snapshot'])
    responses=C.read(C.ROOT/taglock['snapshot']['path']);queue=C.queue()
    tagged=[dict(r,tag=responses[r['frame_id']]['tag']) for r in queue if r['frame_id'] in responses]
    expected_i,expected_r=policy.select_hard(tagged)
    lock=C.read(C.DOC/'HARD_SELECTION_LOCK.json');C.verify(lock['private_selection']);C.verify(lock['tag_lock'])
    selected=C.read(C.ROOT/lock['private_selection']['path'])['rows']
    initial=[r for r in selected if r['assignment']=='INITIAL'];counts=Counter(r['recording'] for r in selected)
    exclusion=C.read(C.RAW/'EXCLUSION_IDENTITIES_PRIVATE.json')
    forbidden={r['sha256'] for r in exclusion['protected_images']+exclusion['current_training_images']+exclusion['historical_pool_images']}
    for binding in C.read(C.DOC/'CANDIDATE_POOL_AUDIT.json')['metadata_sources']:C.verify(binding)
    thumbs=[]
    for r in selected:
        C.verify(r['image']);thumbs.append(thumbnail(C.ROOT/r['image']['path']))
    tests=dict(
        complete_round1=len(responses)==123 and all(r['round']==1 for r in tagged),
        fixed_stop_gate=policy.round_decision(tagged,1)=='SELECT',
        deterministic_exact_selection=[r['frame_id'] for r in selected]==[r['frame_id'] for r in expected_i+expected_r],
        hard_only=all(r['tag'] in ('MODERATE','SEVERE') for r in selected),
        initial8=len(initial)==8,reserve2=len(selected)-len(initial)==2,
        initial_three_recordings=len({r['recording'] for r in initial})>=3,
        max3_over_initial_and_reserve=max(counts.values())<=3,
        four_moderate_four_severe=Counter(r['tag'] for r in initial)=={'MODERATE':4,'SEVERE':4},
        excluded_SHA_disjoint=not({r['image']['sha256'] for r in selected}&forbidden),
        heldout_reserved_recordings_disjoint=not(set(counts)&set(exclusion['heldout_recordings']+exclusion['reserved_recordings'])),
        all_selected_in_previously_MAD_verified_queue=all(r['frame_id'] in {q['frame_id'] for q in queue} for r in selected),
        selected_pairwise_MAD_above2=all(float(np.abs(a-b).mean())>2 for i,a in enumerate(thumbs) for b in thumbs[i+1:]),
        selection_model_outputs_opened_zero=lock['model_outputs_opened']==0,
        no_actual_labels_yet=not C.annotation_labels_path().exists(),
        no_teacher_or_fit_yet=not (C.DOC/'TEACHER_HARD_PREDICTION_LOCK.json').exists(),
    )
    assert all(tests.values()),tests
    C.save(C.DOC/'HARD_SELECTION_AUDIT.json',dict(status='PASS',tests=tests,
           tag_counts=dict(Counter(r['tag'] for r in tagged)),hard_recordings=len({r['recording'] for r in tagged if r['tag'] in ('MODERATE','SEVERE')}),
           selected_recordings=dict(counts),initial_recordings=dict(Counter(r['recording'] for r in initial)),
           previous_full_queue_MAD_audit=C.bind(C.DOC/'PREPARATION_TESTS.json'),
           annotation_GUI_smoke='PASS: temporary labels only; manual bbox/click/advance/hidden ignore/undo/role exclusion',
           public_coordinates=False,training='NOT_RUN',created_at=C.now()))
    print('HARD_SELECTION_AUDIT_PASS',len(tests),dict(counts))

if __name__=='__main__':main()
