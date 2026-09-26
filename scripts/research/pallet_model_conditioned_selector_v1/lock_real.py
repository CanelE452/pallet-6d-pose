"""GT-free real decision process: lock all eight combinations before evaluation reads."""
from . import common as C
RGB_READ_ALLOWLIST=set()
READS=C.guard('decision',RGB_READ_ALLOWLIST)
import numpy as np
import torch
import cv2
from scripts.research.pallet_selector_recovery_v1 import features as F,models as M

def main():
    assert not (C.stage(3)/'REAL_SELECTOR_DECISION_LOCK.json').exists(),'Decisions already frozen'
    lock=C.read(C.stage(2)/'SCORER_LOCK.json');b=C.read(C.DOC/'INPUT_BINDINGS.json')
    bindings={'OLD_GEO':b['old_scorer'],'S1SPEC_GEO':lock['checkpoints']['S1'],'HMANSPEC_GEO':lock['checkpoints']['H_MANUAL']}
    for v in bindings.values():C.verify(v)
    ck={k:torch.load(C.ROOT/v['path'],map_location='cpu',weights_only=False) for k,v in bindings.items()}
    pl=C.read(C.HARD/'POSE_DECISIONS_LOCK.json');rl=C.read(C.HARD/'RAW_PREDICTIONS_LOCK.json');C.verify(pl['poses']);C.verify(rl['predictions']);C.verify(rl['metadata'])
    rows=C.read(C.ROOT/rl['metadata']['path'])['real'];raw=C.read(C.ROOT/rl['predictions']['path'])['real'];poses=C.read(C.ROOT/pl['poses']['path'])['real'];out={};features={};parity=[]
    assert len(rows)==128
    RGB_READ_ALLOWLIST.update(str((C.ROOT/r['image']['path']).absolute()) for r in rows)
    # Original hard experiment stored native RGB dimensions separately in memory.
    # Recover only image shape, not annotation/GT, for the identical area feature.
    for r in rows:
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
        r['hw']=list(im.shape[:2])
    for model in C.MODELS:
        a=C.ARM[model];features[model]={}
        for s in C.SELECTORS:out[model+'_'+s]={}
        for r in rows:
            fid=r['id'];f=F.extract(raw[a][fid],r['K'],r['dims'],r['hw']);features[model][fid]=f
            old=poses[a][fid]
            for h in f['hypotheses']:
                p=F.production_pose(h,r['dims']);saved=next(q['pose'] for q in old['hypotheses'] if q['name']==h['name'])
                assert p.keys()==saved.keys()
                for k in p:
                    if isinstance(p[k],list):np.testing.assert_allclose(p[k],saved[k],atol=1e-7,rtol=1e-7)
                    else:assert p[k]==saved[k]
            assert f['selection']==old['D9'].get('selected_hypothesis')
            for selector in C.SELECTORS:
                selected=f['selection'];scores=None;fallback=None
                if selector=='D9':
                    scores=[h.get('score') for h in f['hypotheses']]
                elif f['valid']:
                    scores=M.scores(ck[selector],np.array(f['features'],np.float32)[None])[0];idx=M.selection(scores[None],C.HYP)[0]
                    assert idx==1-M.selection(scores[None,::-1],C.HYP[::-1])[0];selected=C.HYP[idx]
                else:fallback='invalid feature pair: retain unchanged production D9'
                if selector=='OLD_GEO':assert selected==old['selected'];parity.append(fid+'/'+model)
                margin=abs(float(scores[0]-scores[1])) if scores is not None and len(scores)==2 and all(v is not None and np.isfinite(v) for v in scores) else None
                out[model+'_'+selector][fid]=dict(model=model,selector=selector,scores=scores,selected=selected,score_margin=margin,
                    fallback_reason=fallback,valid_pair=f['valid'],valid_pose=any(h['name']==selected and F.production_pose(h,r['dims'])['available'] for h in f['hypotheses']))
    C.save(C.RAW/'REAL_SELECTOR_DECISIONS_PRIVATE.json',out);C.save(C.RAW/'REAL_FEATURES_PRIVATE.json',features)
    C.save(C.stage(3)/'REAL_SELECTOR_DECISION_LOCK.json',dict(created_at=C.now(),GT_input=False,decisions=C.bind(C.RAW/'REAL_SELECTOR_DECISIONS_PRIVATE.json'),
        features=C.bind(C.RAW/'REAL_FEATURES_PRIVATE.json'),scorer_lock=C.bind(C.stage(2)/'SCORER_LOCK.json'),scorers=bindings,raw_prediction_lock=C.bind(C.HARD/'RAW_PREDICTIONS_LOCK.json'),
        combos=list(out),frames=128,decisions_count=sum(map(len,out.values())),same_candidates_as_original=True,old_selection_parity_count=len(parity),
        candidate_order_swap=True,read_paths=sorted(set(READS)),no_severity_routing=True,weights_unchanged=True))
    print('REAL_EIGHT_COMBINATIONS_LOCKED',1024,flush=True)

if __name__=='__main__':main()
