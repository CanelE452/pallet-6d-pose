"""Preparation-stage invariants. Future labeling/scoring gates are explicitly pending."""
import copy
import ast
from collections import Counter
import cv2
import numpy as np
from . import common as C
from .prepare import select

def main():
    checks={}
    def check(name,condition):
        assert condition,name
        checks[name]='PASS'
    s=C.read(C.RAW/'ANCHOR_SELECTION.json');split=C.read(C.SPLIT)
    check('anchor_from_heldout_only',set(r['frame_id'] for r in s['frames'])<=set(r['id'] for r in split['heldout']))
    check('train_recordings_excluded',not set(r['recording'] for r in s['frames'])&{'REC_001','REC_002'})
    check('balanced_18',Counter(r['severity'] for r in s['frames'])=={v:6 for v in C.SEVERITIES})
    check('model_outputs_not_used_for_selection',s['model_output_used'] is False and all(r['no-model-output-used'] for r in s['frames']))
    check('selection_fixed_before_annotation',C.sha(C.RAW/'ANCHOR_SELECTION.json')==C.read(C.DOC/'INPUT_BINDINGS.json')['selection_sha256'])
    pool=[dict(frame_id=r['id'],severity=r['severity'],recording=r['recording_group'],image=r['image']) for r in split['heldout']]
    thumbs={}
    for r in pool:
        p=C.ROOT/r['image']['path'];assert C.sha(p)==r['image']['sha256']
        im=cv2.imread(str(p));r['hw']=list(im.shape[:2]);thumbs[r['frame_id']]=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(64,48)).astype(np.float32)
    check('image_hashes',True)
    a,b=select(pool,thumbs)
    check('deterministic_selection', [r['frame_id'] for r in a]==[r['frame_id'] for r in s['frames']] and b==s['relaxations'])
    check('no_near_duplicate_anchor_pair', all(float(np.abs(thumbs[r['frame_id']]-thumbs[q['frame_id']]).mean())>2 for i,r in enumerate(a) for q in a[:i]))
    current=C.contract();frozen=C.read(C.DOC/'CORNER_CONTRACT.json')
    import json
    check('corner_contract_matches_repo',json.loads(json.dumps(current))==frozen)
    check('contract_edges_consistent',len(frozen['edges'])==12 and len(frozen['xyz_m'])==8 and frozen['xyz_m'][0][2]<frozen['xyz_m'][4][2])
    for st in C.STATUSES:
        check(f'status_{st}', C.valid_corner(dict(status=st,xy=[10.,10.]),[480,640])==(st in ('DIRECT_VISIBLE','VIRTUAL_INFERABLE')))
        check(f'empty_{st}', C.valid_corner(dict(status=st,xy=None),[480,640])==(st!='DIRECT_VISIBLE'))
    check('bad_coordinates_rejected',all(not C.valid_corner(dict(status='DIRECT_VISIBLE',xy=q),[480,640]) for q in ([640,0],[-1,0],[0,480],[float('nan'),0],[True,0])))
    template=C.read(C.RAW/'LABEL_TEMPLATE.json')
    check('unreviewed_not_uncertain',all(c['status'] is None and c['xy'] is None for r in template['frames'] for c in r['corners']))
    check('initial_progress_zero',C.progress(template,s)['completed_statuses']==0)
    fake=copy.deepcopy(template)
    for r in fake['frames']:
        for c in r['corners']:c['status']='UNCERTAIN'
    check('status_only_completion',C.progress(fake,s)['completed_images']==18)
    ui=(C.ROOT/'scripts/research'/C.NAME/'label_anchor.py').read_text()
    # Explicit allowlist of files exposed by UI. No data-dependent glob / remote assets.
    check('first_pass_no_prediction_overlay',not any(t in ui for t in ('TRUTH_FOR_DISPLAY_ONLY','REFERENCE_PREDICTIONS','YOLO','torch','annotation\'','annotation"','legacy_ref')))
    for module in ('prepare.py','common.py','label_anchor.py','validate_labels.py'):
        tree=ast.parse((C.ROOT/'scripts/research'/C.NAME/module).read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)] + [v.name for n in ast.walk(tree) if isinstance(n,ast.Import) for v in n.names]
        check('no_training_dependencies_'+module,not any(x.startswith(('torch','ultralytics','scripts.research.pallet_existing_data_transfer')) for x in imports))
    check('existing_split_unchanged',C.sha(C.SPLIT)==s['source_split_sha256'])
    import re
    for md in C.DOC.glob('*.md'):
        for link in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',md.read_text()):assert (md.parent/link).exists()
    check('report_images_resolve',True)
    result=dict(checks=checks,stage='PREPARATION_ONLY',pending=['first_pass_lock_before_legacy_compare','QA_not_model_selected',
        'verified_labels_not_train','predictions_lock_before_scoring','fixed_identity_primary','no_hidden_primary','post_label_GT_preservation'],
        no_optimizer_created=True,no_training=True,no_inference=True)
    destination=C.DOC/'PREPARATION_TESTS.json'
    if destination.exists():
        assert C.read(destination)==result, 'Test result changed; preserve existing audit and inspect'
    else:
        C.save_new(destination,result)
    print('PASS',len(checks),'preparation checks; human/scoring checks pending')

if __name__=='__main__':main()
