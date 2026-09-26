"""Phase1–2 audit. Human/fit/eval tests are NOT_RUN, never fabricated PASS."""
import ast
from collections import Counter
import unittest
from . import common as C
from .prepare import thumbnail,MADIndex
from .policy import temporal_queue

def main():
    b=C.read(C.DOC/'INPUT_BINDINGS.json')
    for entry in b['files']:C.verify(entry)
    a=C.read(C.DOC/'CANDIDATE_POOL_AUDIT.json');C.verify(a['private_inventory'])
    rows=C.read(C.RAW/'CANDIDATE_POOL_PRIVATE.json')['rows'];queue=C.queue()
    exclusions=C.read(C.RAW/'EXCLUSION_IDENTITIES_PRIVATE.json')
    forbidden={v['sha256'] for v in exclusions['protected_images']+exclusions['current_training_images']+exclusions['historical_pool_images']}
    held=set(exclusions['heldout_recordings']);reserved=set(exclusions['reserved_recordings'])
    sha={r['image']['sha256'] for r in rows}
    tests={
      'head_recorded':len(b['head'])==40,
      'input_hashes_unchanged':True,
      'adaptation_pool_only':all('/data/pallet/raw_data/' in str(C.ROOT/r['image']['path']) for r in rows),
      'eval_anchor_clean10_h10_training_final_SHA_excluded':not(sha&forbidden),
      'heldout_and_reserved_recordings_excluded':not({r['recording'] for r in rows}&(held|reserved)),
      'unique_pool_SHA':len(sha)==len(rows),
      'fixed_temporal_queue_reproducible':queue==temporal_queue(rows),
      'round_order':all(r['round']==r['bin']%3+1 for r in queue),
      'max16_per_recording_per_round':max(Counter((r['recording'],r['round']) for r in queue).values())<=16,
      'model_output_values_not_read_in_inventory':a['model_outputs_opened']==0,
      'coordinate_GT_not_read_in_inventory':a['coordinate_GT_opened']==0,
      'no_human_tag_fabrication':not (C.RAW/'DIFFICULTY_TAGS_PRIVATE.json').exists(),
      'training_not_run':not (C.DOC/'RAW_PREDICTIONS_LOCK.json').exists(),
    }
    # Independent full-queue recomputation against ALL protected thumbnails.
    index=MADIndex(len(exclusions['protected_images'])+len(queue))
    for i,entry in enumerate(exclusions['protected_images']):
        C.verify(entry);index.add(thumbnail(C.ROOT/entry['path']),entry['sha256'])
        if i%1500==0:print('AUDIT_PROTECTED',i,flush=True)
    for r in queue:
        C.verify(r['image']);t=thumbnail(C.ROOT/r['image']['path'])
        assert index.match(t) is None, r['frame_id']
        index.add(t,r['frame_id'])
    tests['full_queue_near_duplicate_zero_against_protected_and_each_other']=True
    # Check GUI and sampling import graphs do not contain model/scoring helpers.
    modules=[]
    for name in ('prepare.py','policy.py','tag_difficulty.py','annotate_hard.py','labels.py'):
        tree=ast.parse((C.ROOT/'scripts/research'/C.NAME/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):modules.extend(a.name for a in node.names)
            if isinstance(node,ast.ImportFrom):modules.append(node.module or '')
    tests['no_model_import_in_candidate_or_GUI']=not any(any(s in x for s in ('torch','ultralytics','evaluate','posefix','teacher')) for x in modules)
    tests['no_public_private_coordinates']=all(not any(k in C.read(p) for k in ('xy','keypoints_xy','bbox')) for p in C.DOC.glob('*.json'))
    suite=unittest.defaultTestLoader.loadTestsFromName('scripts.research.pallet_min_hard_ab_v1.test_workflow')
    result=unittest.TextTestRunner(verbosity=1).run(suite);tests['unit_tests']=result.wasSuccessful()
    assert all(tests.values()),tests
    C.save(C.DOC/'PREPARATION_TESTS.json',dict(status='PASS_PHASE1_2',tests=tests,unit_tests=result.testsRun,
           upstream_hashes=len(b['files']),full_queue_MAD_rechecked=len(queue),created_at=C.now(),
           NOT_RUN=['actual human tag distribution','hard selection/annotation','teacher inference',
                    'fit pair integrity','320step H fits','prediction/pose locks','new A/B evaluation']))
    print('PASS_PHASE1_2',len(tests),'checks +',result.testsRun,'unit tests')

if __name__=='__main__':main()
