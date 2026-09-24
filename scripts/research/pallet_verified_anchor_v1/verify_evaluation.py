"""Independent arithmetic/provenance checks of the completed visible evaluation."""
import ast
import math
import re
import statistics
from . import common as C
from .review_saved_keypoints import REVIEW
from .evaluate import metrics,point

def main():
    checks={}
    def check(name,value):
        assert value,name
        checks[name]='PASS'
    labels=C.read(REVIEW/'VERIFIED_LABELS.json')['labels'];sel=C.read(REVIEW/'ANCHOR_SELECTION.json')
    rows=C.read(REVIEW/'SCORED_POINTS_PRIVATE.json');res=C.read(C.DOC/'VERIFIED_RESULTS.json')
    lock=C.read(REVIEW/'FIRST_PASS_LOCK.json')
    check('status_review_unchanged_after_lock',C.sha(REVIEW/'LABELS.json')==lock['labels_sha256'])
    check('first_pass_preserved',labels==C.read(REVIEW/'FIRST_PASS_SNAPSHOT.json'))
    check('all_72_requested_statuses_complete',len(labels['review_queue'])==72 and all(labels['frames'][i]['corners'][j]['status'] in C.STATUSES for i,j in labels['review_queue']))
    check('no_QA_skipped',not C.read(REVIEW/'QA_QUEUE.json')['points'])
    check('visible_only_66',len(rows)==66)
    by={s['frame_id']:i for i,s in enumerate(sel['frames'])}
    for row in rows:
        c=labels['frames'][by[row['frame_id']]]['corners'][row['corner_id']]
        assert c['status']=='DIRECT_VISIBLE' and c['coordinate_source']=='manual_click'
        assert row['verified_xy']==c['xy'] and C.valid_corner(c,sel['frames'][by[row['frame_id']]]['hw'])
    check('PnP_hidden_extrapolated_outside_not_primary',True)
    plock=C.read(C.DOC/'PREDICTIONS_LOCK.json')
    for b in plock['files']:assert C.sha(C.ROOT/b['path'])==b['sha256']
    check('predictions_frozen_and_verified_binding',plock['verified_labels_sha256']==C.sha(REVIEW/'VERIFIED_LABELS.json'))
    check('identity_not_remapped',res['fixed_identity'] and not res['symmetry_remapping'])
    for group,arms in res['groups'].items():
        rr=[r for r in rows if group=='ALL' or r['severity']==group or r['recording']==group]
        for arm,m in arms.items():
            values=[math.dist(r['verified_xy'],r['model_xy'][arm]) for r in rr]
            assert all(abs(v-r['errors'][arm])<1e-9 for v,r in zip(values,rr))
            assert len(values)==m['n'] and abs(statistics.median(values)-m['median_px'])<1e-9
            for threshold in (5,10,20):assert sum(e<=threshold for e in values)==m['PCK'][str(threshold)]['correct']
    check('independent_fixed_identity_arithmetic',True)
    check('inclusive_PCK_threshold',metrics([5.,10.,20.,20.01])['PCK']['20']['correct']==3)
    check('missing_prediction_is_not_fake_point',point({'selected_index':None},0) is None)
    old=C.read(C.ROOT/'outputs/pallet_verified_anchor_v1/keypoints_first/WORKSPACE.json')
    for r in old['frames']:
        for key in ('image','source_annotation'):assert C.sha(C.ROOT/r[key]['path'])==r[key]['sha256']
    check('original_GT_and_RGB_unchanged',True)
    for b in C.read(REVIEW/'INPUT_LOCK.json')['annotations']:assert C.sha(C.ROOT/b['path'])==b['sha256']
    check('user_keypoint_files_unchanged',True)
    for name in ('audit_completed_status.py','evaluate.py','report.py'):
        tree=ast.parse((C.ROOT/'scripts/research'/C.NAME/name).read_text())
        names=[v.name for n in ast.walk(tree) if isinstance(n,ast.Import) for v in n.names]
        names += [n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not any(n.startswith(('torch','ultralytics')) for n in names)
    check('no_training_or_inference_dependencies',True)
    report=C.DOC/'EVALUATION_REPORT_KO.md'
    imgs=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report.read_text())
    check('all_26_report_images_resolve',len(imgs)==26 and all((report.parent/p).exists() for p in imgs))
    check('no_more_anchor_images',C.read(C.DOC/'FINAL_DECISION.json')['additional_anchor_images']==0)
    result=dict(checks=checks,verified_points=66,new_training=0,new_inference=0,
        limitation='Human annotation and PnP assistance are not independently verified physical ground truth')
    path=C.DOC/'EVALUATION_TESTS.json'
    if path.exists():assert C.read(path)==result, 'Preserve prior audit; inspect changed checks'
    else:C.save_new(path,result)
    print('PASS',len(checks),'completed evaluation checks')

if __name__=='__main__':main()
