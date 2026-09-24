"""Freeze existing raw predictions then production D9 poses. NO truth inputs."""
from pathlib import Path
import numpy as np
from . import common as C

def main():
    assert not (C.DOC/'PREDICTIONS_LOCK.json').exists(), 'Frozen output already exists'
    pair=C.read(C.DOC/'PAIR_INTEGRITY.json'); assert pair['status']=='CAUSAL_PAIR_VERIFIED'
    bindings=C.read(C.DOC/'INPUT_BINDINGS.json')['files']
    ids=[r['id'] for r in C.records()]
    assert len(ids)==128 and len(set(ids))==128
    files=[C.E.RAW/'REFERENCE_PREDICTIONS.json',*[C.E.C.RAW/f'EVAL_PLASTIC_{a}.json' for a in ('S0','S1')]]
    for p in files:
        C.verify(next(b for b in bindings if b['path']==str(p.relative_to(C.ROOT))))
    refs=C.read(files[0]);preds={'R0':refs['R0']}
    for a,p in zip(('S0','S1'),files[1:]):
        obj=C.read(p)
        C.verify(obj['checkpoint']);assert not obj['GT_input']
        preds[a]={i:obj['predictions'][i] for i in ids}
    assert preds['S1']==refs['OLD_S1']
    assert all(set(p)==set(ids) for p in preds.values())
    C.save(C.RAW/'PREDICTIONS.json',preds)
    C.save(C.DOC/'PREDICTIONS_LOCK.json',dict(file=C.bind(C.RAW/'PREDICTIONS.json'),created_at=C.now(),
        before_scoring=True,GT_input=False,cache_reused=True,new_raw_inference=0,frames=128,
        args=dict(conf=.001,imgsz=640,rect=True,padding=100,padding_mode='BORDER_REFLECT_101',augment=False,half=False,selected='argmax confidence'),
        sources=[C.bind(p) for p in files]))
    meta={r['id']:r for r in C.read(C.E.V.RAW/'INFERENCE_METADATA.json')}
    poses={}
    for a,pp in preds.items():
        poses[a]={}
        for fid in ids:
            q=C.E.D.points(pp[fid]);m=meta[fid];K=np.array(m['K']);dims=np.array(m['xyz'])
            sel=C.E.D.select(q,K,dims)
            current=C.E.D.Pose.infer(q,K,dims,False)
            assert current.get('selected_hypothesis')==sel.get('selected_hypothesis')
            hypotheses=[dict(name=h['name'],score=h['score'],pose=C.E.D.hyp_pose(h,q,K,dims)) for h in sel['hypotheses']]
            selected=next((h for h in hypotheses if h['name']==sel.get('selected_hypothesis')),None)
            if selected: C.E.D.close(selected['pose'],current)
            poses[a][fid]=dict(current=current,hypotheses=hypotheses)
        print('D9_POSE_FROZEN',a,len(poses[a]),flush=True)
    C.save(C.RAW/'POSE_PREDICTIONS.json',poses)
    C.save(C.DOC/'POSE_PREDICTIONS_LOCK.json',dict(file=C.bind(C.RAW/'POSE_PREDICTIONS.json'),created_at=C.now(),before_scoring=True,
        GT_input=False,production_D9_unchanged=True,selector=C.bind(Path(C.E.D.Selector.__file__)),solver=C.bind(Path(C.E.D.Pose.__file__)),
        metadata=C.bind(C.E.V.RAW/'INFERENCE_METADATA.json'),predictions_lock=C.bind(C.DOC/'PREDICTIONS_LOCK.json')))

if __name__=='__main__': main()
