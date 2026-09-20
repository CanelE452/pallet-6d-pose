"""One fixed IoU90 student, reusing the original trainer and paper evaluators.

Path overrides are scoped to this process; previous source/results stay immutable.
"""
import argparse
import copy
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import common as C
from . import train as T
from . import evaluate as E
from . import paper_metrics_plastic as M
from .pseudo import top

BASE_RAW=C.RAW
BASE_DOC=C.DOC
RAW=BASE_RAW/'iou_selftrain'
DOC=BASE_DOC/'iou_selftrain'
ARM='IOU90'


@contextmanager
def scope():
    with patch.object(C,'RAW',RAW),patch.object(C,'DOC',DOC),patch.object(M,'RAW',RAW/'metrics'),patch.object(M,'DOC',DOC/'metrics'):
        yield


def selected_ids(result):
    return {r['id'] for r in result['records'] if r['correction']>=.9 and r['augmentation']>=.9}


def prepare():
    parent=C.read(BASE_DOC/'TRAIN_PROTOCOL.json')
    for b in parent['inputs']+parent['sources']+[parent['code'],parent['common'],parent['test_code'],parent['initialization']]:C.verify(b)
    audit=C.read(BASE_RAW/'iou_review/RESULTS.json');ids=selected_ids(audit);assert len(ids)==167
    assert ids=={r['id'] for r in audit['records'] if r['id'] not in audit['thresholds']['0.9']['either']['rejected_ids']}
    candidates=[r for r in C.read(BASE_RAW/'PSEUDO_ACCEPTED.json') if r['kind']=='PLASTIC' and r['id'] in ids]
    assert len(candidates)==167
    old_slots=Path(C.ROOT/parent['datasets']['PLASTIC']['train_list']['path']).read_text().splitlines()
    old_syn=old_slots[:512];assert len(set(old_syn))==512 and all(Path(x).name.startswith('syn__') for x in old_syn)
    shared=RAW/'dataset';bindings=[];new_syn=[];new_real=[]
    for old in old_syn+[str(BASE_RAW/'dataset/images'/f'{r["id"]}.png') for r in candidates]:
        image=Path(old);label=image.parent.parent/'labels'/image.with_suffix('.txt').name
        assert image.exists() and label.exists()
        new=shared/'images'/image.name;new_label=shared/'labels'/label.name
        T.link(image,new);T.link(label,new_label)
        assert C.sha(new)==C.sha(image) and C.sha(new_label)==C.sha(label)
        bindings.extend([C.bound(new),C.bound(new_label)])
        (new_syn if image.name.startswith('syn__') else new_real).append(str(new))
    # Keep the exact prior replacement sampling policy and seed, report actual unique exposure.
    replacement=np.random.default_rng(9021).choice(sorted(new_real),512,replace=True).tolist()
    slots=new_syn+replacement;assert len(slots)==1024
    with scope():
        C.write_text(shared/'train.txt','\n'.join(slots)+'\n')
        C.write_text(shared/'val.txt',(BASE_RAW/'dataset/val.txt').read_text())
        C.write_text(shared/'data.yaml',f'path: {shared}\ntrain: {shared/"train.txt"}\nval: {shared/"val.txt"}\nnc: 1\nnames: [pallet]\nkpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n')
        p=copy.deepcopy(parent)
        p.update(arms=[ARM],datasets={ARM:dict(data=C.bound(shared/'data.yaml'),train_list=C.bound(shared/'train.txt'),
            slots=1024,pseudo_slots=512,pseudo_unique=167,sampled_pseudo_unique=len(set(replacement)))},
            inputs=parent['inputs']+bindings,extension_code=C.bound(__file__),
            change='Only eligible real pool changes: both correction and augmentation hull IoU >=0.9. Same R0 init, optimizer, args, synthetic512, replacement sampling seed9021,5epochs,320steps.',
            manual_review='Not used; no human decision JSON or corner exclusion applied.',
            parent_protocol=C.bound(BASE_DOC/'TRAIN_PROTOCOL.json'))
        p['sources']+=[C.bound(BASE_RAW/'iou_review/RESULTS.json'),C.bound(BASE_DOC/'iou_review/PROTOCOL.json'),C.bound(__file__)]
        assert p['args']==T.ARGS==parent['args']
        C.freeze(DOC/'TRAIN_PROTOCOL.json',p)
        C.freeze(RAW/'SELECTED_UNCHANGED_PSEUDOLABELS.json',candidates)
        assert all(T.label(top(r['refined']),r['raw_hw'])==(shared/'labels'/f'{r["id"]}.txt').read_text() for r in candidates)
        ev=C.read(BASE_DOC/'EVAL_PROTOCOL.json');ev['records']=[r for r in ev['records'] if r['kind']=='PLASTIC'];ev['arms']=[ARM]
        assert len(ev['records'])==194
        assert not ({r['image']['sha256'] for r in ev['records']} & {r['image']['sha256'] for r in candidates})
        ev['sources']+=[C.bound(DOC/'TRAIN_PROTOCOL.json'),C.bound(__file__)]
        ev['primary']='Same complete plastic194; no evaluation-time filtering or refiner; fixed last checkpoint. Compare R0 and original unfiltered-pool student.'
        C.freeze(DOC/'EVAL_PROTOCOL.json',ev)
        met=copy.deepcopy(C.read(BASE_DOC/'paper_metrics_plastic/PROTOCOL.json'))
        met['arms']=[ARM];met['sources']+=[C.bound(DOC/'TRAIN_PROTOCOL.json'),C.bound(__file__)]
        C.freeze(DOC/'metrics/PROTOCOL.json',met)
        C.freeze(DOC/'DATA_AUDIT.json',dict(selected_pseudo_images=167,sampled_unique=len(set(replacement)),
            pseudo_slots=512,synthetic_slots=512,epochs=5,optimizer_steps_expected=320,
            same_synthetic_images=True,same_exported_label_bytes=True,no_eval_image_overlap=True,
            parent_training_verification=C.bound(BASE_DOC/'DATA_VERIFICATION.json'),
            train_list=C.bound(shared/'train.txt'),selected_labels=C.bound(RAW/'SELECTED_UNCHANGED_PSEUDOLABELS.json')))
    print('PREPARED',167,'eligible;',len(set(replacement)),'unique sampled;512+512 slots,5epochs,320steps',flush=True)


def score():
    for b in C.read(DOC/'TRAIN_PROTOCOL.json')['sources']:C.verify(b)
    for b in C.read(BASE_DOC/'paper_metrics_plastic/FINAL_AUDIT.json')['artifacts']:C.verify(b)
    with scope():
        # Exact image/annotation identity mapping from the already verified archived pose bridge.
        mapping=C.read(BASE_DOC/'paper_metrics_plastic/POSE_ID_BRIDGE.json')['mapping']
        original=M.positive
        def pose_positive(arm):
            payload,ps=original(arm);assert set(ps)==set(mapping)
            return payload,{mapping[k]:v for k,v in ps.items()}
        f=M.f0(ARM);d=M.detection(ARM)
        with patch.object(M,'positive',pose_positive):pose=M.pose(ARM)
        prior=C.read(BASE_RAW/'paper_metrics_plastic/SUMMARY.json')['arms']
        result=dict(arms={**prior,ARM:dict(f0=f,pose=pose,detection=d)},
            training=C.read(DOC/'DATA_AUDIT.json'),fit=C.read(DOC/f'FIT_{ARM}.json'),
            selected_labels=C.bound(RAW/'SELECTED_UNCHANGED_PSEUDOLABELS.json'),
            population='Same ordinary plastic194 + negative2689, no evaluation filtering',
            independent_confirmation=False,auto_promoted=False,single_seed=True)
        # Supplementary full-denominator symmetry-aware PCK from prior experiment, not paper kp columns.
        pe,pop=E.O.population_metadata()
        from scripts.research.pallet_dim_conditioned_p_v1 import eval_math
        groups={r['object_type']:r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']}
        targets={item.frame_id:pe.E._legacy_forbidden_target(item) for item,meta in pop if meta['object_type']==C.TYPES['PLASTIC']}
        records=[]
        for row in C.read(RAW/f'EVAL_PREDICTIONS_{ARM}.json')['records']:
            t=targets[row['id']];p=top(row['prediction']);matched=p is not None and E.O.iou(p['box_xyxy'],t.box_xyxy)>=.5
            q=np.full((9,2),np.nan) if p is None else p['keypoints_xy']
            v=eval_math.measure(q,np.asarray(t.keypoints_xy),np.asarray(t.keypoint_supervision_mask),groups[C.TYPES['PLASTIC']],row['raw_hw'],matched,p is not None)
            records.append(dict(id=row['id'],kind='PLASTIC',**v))
        C.freeze(RAW/'SYMMETRY_METRICS.json',dict(summary=eval_math.summary(records),records=records))
        result['symmetry']=eval_math.summary(records)
        C.freeze(RAW/'SUMMARY.json',result)
        for a,r in result['arms'].items():
            m=r['detection']['report']['metrics']['box_and_keypoint_2d']
            print(a,'AP',m['box_ap50_95'],'kpmed',m['keypoint_location_median_px'],'FPR95',r['detection']['ranking']['fpr95'],'pose',r['pose'],flush=True)
        print('PCK',result['symmetry'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','eval','negative','score']);a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='score':score()
    else:
        with scope():
            if a.phase=='train':T.train(ARM)
            elif a.phase=='eval':E.infer(ARM)
            else:M.infer_negative(ARM)
